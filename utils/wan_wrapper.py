import types
from contextlib import nullcontext
from typing import List, Optional, Tuple
import torch
from torch import nn

import time
from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
import os
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader

# from settings import MODEL_FOLDER
MODEL_FOLDER = None  # Set via config: text_encoder_path / vae_path, or wan_model_folder


from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_folder: str) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('meta')
        ).eval().requires_grad_(False)
        self.text_encoder.to_empty(device='cpu')

        safetensors_path = os.path.join(model_folder, "models_t5_umt5-xxl-enc-bf16.safetensors")
        self.text_encoder.load_state_dict(safe_load_file(safetensors_path))

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_folder, "google", "umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        stime = time.time()
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0
        result = { "prompt_embeds": context }
        return result


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_folder: str):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        vae_path = os.path.join(model_folder, "Wan2.1_VAE.pth")
        self.model = _video_vae(
            pretrained_path=vae_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def forward(self, x: torch.Tensor, method: str = 'encode', **kwargs) -> torch.Tensor:
        if method == 'encode':
            return self.encode_to_latent(x)
        elif method == 'decode':
            return self.decode_to_pixel(x, **kwargs)
        else:
            raise ValueError(f"Unknown method {method}")

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

def load_state_dict_from_folder_safetensors(file_path):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and "diffusion" in file_name and file_name.split(".")[-1] in [
            "safetensors"
        ]:
            state_dict.update(safe_load_file(os.path.join(file_path, file_name)))
    return state_dict

def _filter_state_dict_keys(state_dict, skip_substrings):
    """
    Filter out (do not load) weights whose keys contain any of `skip_substrings`.
    Returns (filtered_state_dict, skipped_keys).
    """
    if not skip_substrings:
        return state_dict, []
    skipped = []
    filtered = {}
    for k, v in state_dict.items():
        if any(s in k for s in skip_substrings):
            skipped.append(k)
            continue
        filtered[k] = v
    return filtered, skipped


def dcp_load_dict(path):
    if path.endswith(".safetensors"):
        auto_state_dict = safe_load_file(path)
        state_dict = {}
        for key, value in auto_state_dict.items():
            # Remove FSDP wrapper prefix if present
            if "._fsdp_wrapped_module." in key:
                key = key.replace("._fsdp_wrapped_module.", ".")
            # Remove model. prefix if present
            if "model." in key:
                key = key.replace("model.", "")
            state_dict[key] = value
        return state_dict

    safe_file_path = path + "/model.safetensors"
    if os.path.exists(safe_file_path):
        state_dict = safe_load_file(safe_file_path)
    else:
        reader = FileSystemReader(path)
        metadata = reader.read_metadata()

        auto_state_dict = {}
        for key, entry in metadata.state_dict_metadata.items():
            auto_state_dict[key] = torch.empty(entry.size, dtype=entry.properties.dtype,device=torch.device('meta'))

        dcp.load(state_dict=auto_state_dict,storage_reader=reader,no_dist=True)
        state_dict = {}
        for key, value in auto_state_dict.items():
            # Remove FSDP wrapper prefix if present
            if "._fsdp_wrapped_module." in key:
                key = key.replace("._fsdp_wrapped_module.", ".")
            # Remove model. prefix if present
            if "model." in key:
                key = key.replace("model.", "")
            state_dict[key] = value
        safe_save_file(state_dict, safe_file_path)
    return state_dict

class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            load_path=None,
            timestep_shift=5.0,
            is_causal=False,
            ckpt_path=None,
            weight_list=[],
            filter_list=[],
            in_dim=36,
            dual_model=False,
            high_noise_threshold=0.5,
    ):
        super().__init__()
        import torch.distributed as dist
        rank = dist.get_rank()
        torch.set_num_threads(32)

        # model_path: use the first weight_list path's directory as the model config source,
        # or fall back to model_name if weight_list is empty
        if weight_list:
            model_path = weight_list[0]['path']
        else:
            model_path = model_name

        # Wan2.2 dual model: config.json is inside high_noise_model/ subdir
        config_path = model_path
        if dual_model and os.path.isdir(os.path.join(model_path, "high_noise_model")):
            config_path = os.path.join(model_path, "high_noise_model") 

        # Initialize primary model
        if is_causal:
            config = CausalWanModel.load_config(config_path)
            config = dict(config)
            config["in_dim"] = in_dim
            with torch.device("meta"):
                self.model = CausalWanModel(**config)
            self.model.to_empty(device='cpu')
        else:
            config = WanModel.load_config(config_path)
            config = dict(config)
            config["in_dim"] = in_dim
            with torch.device("meta"):
                self.model = WanModel(**config)
            self.model.to_empty(device='cpu')

        # Initialize secondary model for dual-model mode (Wan2.2)
        self.model_2 = None
        self.dual_model = dual_model
        self.high_noise_threshold = high_noise_threshold

        if dual_model and not is_causal:
            # Use same config for model_2
            with torch.device("meta"):
                self.model_2 = WanModel(**config)
            self.model_2.to_empty(device='cpu')

        if rank == 0:
            state_dict_full = None
            state_dict_full_2 = None  # For model_2

            if ckpt_path is not None:
                state_dict_full = dcp_load_dict(ckpt_path)
            else:
                for weight_config in weight_list:
                    weight_path = weight_config['path']
                    is_model_2 = weight_config.get('is_model_2', False)

                    # For Wan2.2 dual model: automatically determine high/low noise model
                    # based on directory structure if not explicitly specified
                    if dual_model and not is_causal and 'is_model_2' not in weight_config:
                        # Check if path contains high/low noise indicators
                        if 'high_noise' in weight_path.lower() or 'high' in os.path.basename(weight_path).lower():
                            is_model_2 = False  # high noise -> primary model
                        elif 'low_noise' in weight_path.lower() or 'low' in os.path.basename(weight_path).lower():
                            is_model_2 = True   # low noise -> model_2

                    if os.path.isdir(weight_path):
                        state_dict = load_state_dict_from_folder_safetensors(weight_path)
                    else:
                        state_dict = safe_load_file(weight_path)

                    if is_model_2 and dual_model and not is_causal:
                        # This weight is for model_2 (low noise model in Wan2.2)
                        if state_dict_full_2 is None:
                            state_dict_full_2 = state_dict
                        else:
                            state_dict_full_2.update(state_dict)
                    else:
                        # This weight is for model (primary/high noise model)
                        if state_dict_full is None:
                            state_dict_full = state_dict
                        else:
                            state_dict_full.update(state_dict)

            # Load primary model
            if state_dict_full is not None:
                state_dict_full, _ = _filter_state_dict_keys(state_dict_full, skip_substrings=filter_list)
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_full, strict=False)
                print(f"load_model {model_path} (primary) missing_keys: {len(missing_keys)} unexpected_keys: {len(unexpected_keys)}")

            # Load secondary model (only for dual_model and non-causal mode)
            if dual_model and not is_causal and state_dict_full_2 is not None:
                state_dict_full_2, _ = _filter_state_dict_keys(state_dict_full_2, skip_substrings=filter_list)
                missing_keys_2, unexpected_keys_2 = self.model_2.load_state_dict(state_dict_full_2, strict=False)
                print(f"load_model_2 {model_path} (low noise model for Wan2.2) missing_keys: {len(missing_keys_2)} unexpected_keys: {len(unexpected_keys_2)}")

        dist.barrier()

        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 1560 * 24  # [1, 12 * 2, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model._set_gradient_checkpointing(True)


    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        kv_size: Optional[Tuple[int, int]] = (0, 0),
        image_latent_input: Optional[torch.Tensor] = None,
        render_latent_input: Optional[torch.Tensor] = None,
        freqs_offset: int = 0,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # X0 prediction
        # Handle None inputs for T2V mode
        image_latent_permuted = image_latent_input.permute(0, 2, 1, 3, 4).contiguous() if image_latent_input is not None else None
        render_latent_permuted = render_latent_input.permute(0, 2, 1, 3, 4).contiguous() if render_latent_input is not None else None

        if kv_cache is not None:
            assert(not self.dual_model), "KV cache is not supported for dual-model mode"
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous(),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                kv_size=kv_size,
                image_latent_input=image_latent_permuted,
                render_latent_input=render_latent_permuted,
                freqs_offset=freqs_offset,
            ).permute(0, 2, 1, 3, 4)
            if kv_size[1]<0:
                return flow_pred
        else:
            if self.dual_model:
                assert(self.model_2 is not None), "Model 2 is not loaded"
                normalized_timestep = input_timestep.float() / 1000.0
                use_high_noise = (normalized_timestep >= self.high_noise_threshold).all().item()
                selected_model = self.model if use_high_noise else self.model_2
            else:
                selected_model = self.model

            flow_pred = selected_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                image_latent_input=image_latent_permuted,
                render_latent_input=render_latent_permuted,
                freqs_offset=freqs_offset,
            ).permute(0, 2, 1, 3, 4)


        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def forward_wan22(
        self,
        latent_list: List[torch.Tensor],
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Forward method specifically for Wan2.2 dual-model inference.
        Compatible with T2VAlignedInferencePipeline's direct model call signature.

        Args:
            latent_list: List of latent tensors [B, C, F, H, W]
            t: Timestep tensor [B]
            context: Text embeddings
            seq_len: Sequence length
            **kwargs: Additional arguments

        Returns:
            List of flow predictions
        """
        if not self.dual_model:
            raise ValueError("forward_wan22 is only available for dual-model mode")

        # Select model based on timestep
        normalized_timestep = t.float() / 1000.0
        use_high_noise = (normalized_timestep >= self.high_noise_threshold).all().item()
        selected_model = self.model if use_high_noise else self.model_2

        # Process each latent in the list
        output_list = []
        for latent in latent_list:
            flow_pred = selected_model(
                latent, t=t, context=context, seq_len=seq_len, **kwargs
            )
            output_list.append(flow_pred)

        return output_list

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
