import functools
from wan.modules.attention import attention
import math
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    rope_apply_given_freqs,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import time
import copy
from einops import rearrange

flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
 
class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads, 
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 
        self.qk_norm = qk_norm
        self.eps = eps
        self.fused_projections = False

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        freqs, 
        kv_cache=None,  
        kv_size=(0,0),
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2] 
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x): 
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        roped_query = rope_apply_given_freqs(q, freqs).type_as(v)
        roped_key = rope_apply_given_freqs(k, freqs).type_as(v)

        # print("kv_size",kv_size,"len_x",roped_query.shape[1],"roped_key.shape",roped_key.shape)

        assert kv_cache is not None, "kv_cache must be provided when kv_size > 0" 
        if kv_size[1] < 0:
            len_x = roped_query.shape[1]
            kv_cache["k"][:, kv_size[0]:kv_size[0]+len_x] = roped_key
            kv_cache["v"][:, kv_size[0]:kv_size[0]+len_x] = v
            x = attention(
                roped_query,
                roped_key,
                v
            )
        else:
            if kv_size[1]==0:
                x = attention(roped_query,roped_key,v)
            else:
                x = attention(
                    roped_query,
                    torch.cat([kv_cache["k"][:, kv_size[0]:kv_size[0]+kv_size[1]], roped_key], dim=1),
                    torch.cat([kv_cache["v"][:, kv_size[0]:kv_size[0]+kv_size[1]], v], dim=1)
                )
 
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads, 
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads 
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        freqs_x,
        context,
        context_lens,  
        crossattn_cache=None,
        kv_cache=None,
        kv_size=(0,0),
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """  

        e = (self.modulation + e).chunk(6, dim=1)

        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, freqs_x, kv_cache=kv_cache, kv_size=kv_size)

        x = x + y * e[2]

        # len_x = -3*1560
        # x[:,len_x:] = x[:,len_x:] + self.cross_attn(self.norm3(x[:,len_x:]), context,context_lens, crossattn_cache=crossattn_cache)
        x = x + self.cross_attn(self.norm3(x), context,context_lens, crossattn_cache=crossattn_cache)
        
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        shift, scale = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x =  self.head((self.norm(x) * (1 + scale) + shift))
        
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32, 
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks 
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers 
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        
        self.gradient_checkpointing = False
 
    

    def get_transformer_module(self):
        return {type(self.blocks[0])}

    def init_freqs(self,device):
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        self.freqs = self.freqs.to(device)

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value
 
    def forward(
        self,
        x,
        t,
        context,
        seq_len, 
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None, 
        kv_size=(0,0),
        image_latent_input: torch.Tensor = None,
        render_latent_input: torch.Tensor = None,
        freqs_offset: int = 0,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding 
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device
        if hasattr(self, 'freqs'):
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)
        else:
            self.init_freqs(device)

        f, h, w = x.shape[2:]
        h = h//2
        w = w//2  

        c = self.dim // self.num_heads // 2
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # Compute freqs_x once (same for all branches)
        freqs_x = torch.cat([
            freqs[0][freqs_offset:freqs_offset+f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f*h*w, 1, -1)

        # Concatenate render_latent_input or zeros based on conditions
        if render_latent_input is None:
            assert(x.shape[1] == 16) # t2v
            x = torch.cat([x, torch.zeros_like(x[:, :4]), torch.zeros_like(x[:, :20])], dim=1)
        elif kv_size[1] >= 0:
            assert(x.shape[1] == 16) # v2v
            x = torch.cat([x, render_latent_input], dim=1)
        assert(x.shape[1] == 36) 
        
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([
            torch.as_tensor(u.shape[2:], dtype=torch.long, device=u.device)
            for u in x
        ])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.as_tensor([u.size(1) for u in x], dtype=torch.long, device=x[0].device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # e [1,1536] e0[1,6,1536]
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t[:,0]).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            freqs_x = freqs_x,
            context=context,
            context_lens=None, 
            kv_size=kv_size,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            kwargs['kv_cache'] = kv_cache[block_index]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x= torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x= block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)

        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out