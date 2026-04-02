# InSpatio-World: State Diagrams & LTX-2 v1f Upgrade Plan

> **Phase 1 Implementation Status:** DONE - files created, syntax verified.
> See Section 6 for new file inventory.



## 1. Current Architecture State Diagrams

### 1.1 Full Pipeline Flow (3-Step Orchestration)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  run_test_pipeline.sh                    │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
               ┌──────────────────────────────┼──────────────────────────────┐
               ▼                              ▼                              ▼
    ┌──────────────────┐          ┌──────────────────────┐        ┌──────────────────┐
    │  STEP 1: Caption │          │  STEP 2: Depth+Render│        │  STEP 3: V2V     │
    │  gen_json.py     │          │  convert_da3_to_pi3  │        │  Inference        │
    │                  │          │  render_point_cloud   │        │  inference_causal │
    └────────┬─────────┘          └──────────┬───────────┘        └────────┬─────────┘
             │                               │                             │
             ▼                               ▼                             ▼
    ┌──────────────────┐          ┌──────────────────────┐        ┌──────────────────┐
    │ Florence-2 VLM   │          │ DA3 Depth Estimation │        │ CausalInference  │
    │ Middle frame →   │          │ + Format Conversion  │        │ Pipeline         │
    │ Detailed caption │          │ + Point Cloud Render │        │ (Wan2.1 14B)     │
    └────────┬─────────┘          └──────────┬───────────┘        └────────┬─────────┘
             │                               │                             │
             ▼                               ▼                             ▼
    ┌──────────────────┐          ┌──────────────────────┐        ┌──────────────────┐
    │   OUTPUT:        │          │  OUTPUT:             │        │  OUTPUT:         │
    │   text_prompt    │          │  render_video (RGB)  │        │  novel-view      │
    │   (JSON)         │          │  mask_video (depth)  │        │  video (MP4)     │
    └──────────────────┘          └──────────────────────┘        └──────────────────┘
```

### 1.2 CausalInferencePipeline State Machine

```
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    CausalInferencePipeline.__init__                   │
    │                                                                       │
    │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
    │  │ WanDiffusionWrap │  │ WanTextEncoder   │  │ WanVAEWrapper    │   │
    │  │ (CausalWanModel) │  │ (UM-T5-XXL)      │  │ (Video VAE)      │   │
    │  │ 14B params       │  │ text → [B,512,D]  │  │ pixel ↔ latent   │   │
    │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
    └───────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         .inference() Entry                          │
    │                                                                     │
    │  Inputs:                                                            │
    │    noise:         [B, num_frames, 16, 60, 104]  (random noise)     │
    │    text_prompts:  List[str]                                         │
    │    ref_latent:    [B, F, 16, 60, 104]  (source video latents)      │
    │    render_latent: [B, F, 16, 60, 104]  (depth-rendered guide)      │
    │    mask_latent:   [B, F, 4, 60, 104]   (depth mask)               │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  STATE: INITIALIZE                      │
              │  1. Text encode: prompts → embeddings   │
              │  2. Allocate output tensor               │
              │  3. Initialize/reset KV cache            │
              │     [num_blocks × {k,v}]                │
              │     shape: [B, 9360, heads, head_dim]   │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  STATE: BLOCK LOOP                      │
              │  for block_idx in range(num_blocks):    │
              │    num_frame_per_block = 3               │
              └────────────────────┬───────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  Is this the first block?    │
                    └──────────┬─────────┬────────┘
                         YES   │         │  NO
                               ▼         ▼
              ┌────────────────────┐  ┌────────────────────────┐
              │ context = ref_block│  │ context = ref_block +  │
              │ (source video)     │  │           last_pred    │
              │ kv_size = 1560*3   │  │ kv_size = 1560*3 +     │
              └────────┬───────────┘  │           1560*3       │
                       │              └────────────┬───────────┘
                       └───────────┬───────────────┘
                                   ▼
              ┌────────────────────────────────────────┐
              │  STATE: DENOISE BLOCK                   │
              │  denoise_block() function               │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  SUB-STATE: CONTEXT ENCODING            │
              │  (torch.no_grad)                        │
              │                                         │
              │  generator(                             │
              │    context_frames,                      │
              │    timestep=0,        # clean frames    │
              │    kv_cache=kv_cache, # populates cache │
              │    kv_size=(0, -1),   # append mode     │
              │    freqs_offset=0                       │
              │  )                                      │
              │  → KV cache updated with context keys   │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  SUB-STATE: DENOISING LOOP              │
              │  steps: [1000, 750, 500, 250] (warped)  │
              │                                         │
              │  for step_idx, t in enumerate(steps):   │
              │    ┌─────────────────────────────────┐  │
              │    │ generator(                      │  │
              │    │   noisy_input,                  │  │
              │    │   timestep=t,                   │  │
              │    │   kv_cache, kv_size,            │  │
              │    │   render_block (depth guide),   │  │
              │    │   freqs_offset=6                │  │
              │    │ ) → flow_pred, pred_x0          │  │
              │    └──────────────┬──────────────────┘  │
              │                   │                      │
              │    if not last_step:                     │
              │      noisy_input = scheduler.add_noise(  │
              │        pred_x0, fresh_noise, next_t)     │
              │    else:                                 │
              │      denoised_pred = pred_x0             │
              │                                         │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  STATE: ACCUMULATE                      │
              │  output[:, start:start+F] = pred        │
              │  last_pred = pred.clone().detach()      │
              │  start_index += num_frame_per_block     │
              └────────────────────┬───────────────────┘
                                   │
                          (loop back to BLOCK LOOP)
                                   │
                                   ▼ (all blocks done)
              ┌────────────────────────────────────────┐
              │  STATE: DECODE                          │
              │  video = vae.decode_to_pixel(output)    │
              │  video = (video * 0.5 + 0.5).clamp(0,1)│
              │  → [B, T, 3, H, W] pixel video         │
              └────────────────────────────────────────┘
```

### 1.3 denoise_block() Internal State Machine

```
    denoise_block(generator, scheduler, noisy_input, ...)
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  ┌─────────────┐     ┌─────────────────────────────────┐   │
    │  │ HAS CONTEXT? ├──Y─▶│ CONTEXT ENCODING PASS           │   │
    │  └──────┬──────┘     │ timestep = zeros (clean)         │   │
    │         │ N          │ with torch.no_grad():            │   │
    │         │            │   generator(context, t=0,        │   │
    │         │            │     kv_cache, kv_size=(0,-1))    │   │
    │         │            │ → populates KV cache              │   │
    │         │            └────────────────┬──────────────────┘   │
    │         └────────────────────────────┐│                      │
    │                                      ▼▼                      │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  DENOISING STEPS LOOP                                 │  │
    │  │  steps = [1000, 750, 500, 250] (post-warp)            │  │
    │  │                                                       │  │
    │  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐             │  │
    │  │  │t=1000│→ │t=750 │→ │t=500 │→ │t=250 │             │  │
    │  │  │      │  │      │  │      │  │(LAST) │             │  │
    │  │  │no_grad  │no_grad  │no_grad  │w/grad │             │  │
    │  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘             │  │
    │  │     │         │         │         │                   │  │
    │  │     ▼         ▼         ▼         ▼                   │  │
    │  │  ┌──────────────────────────────────────┐             │  │
    │  │  │ generator(noisy_input, t, kv_cache,  │             │  │
    │  │  │   render_block, freqs_offset=6)      │             │  │
    │  │  │ → flow_pred, pred_x0                 │             │  │
    │  │  └──────────────────────────────────────┘             │  │
    │  │     │         │         │         │                   │  │
    │  │     ▼         ▼         ▼         ▼                   │  │
    │  │  add_noise  add_noise add_noise  DONE                │  │
    │  │  (re-noise  (re-noise (re-noise  (return             │  │
    │  │   @t=750)    @t=500)  @t=250)    denoised)           │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                                                              │
    │  Returns: (denoised_pred, noise_before_last_step)            │
    └──────────────────────────────────────────────────────────────┘
```

### 1.4 Data Flow Through Models

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     ENCODER SIDE (preprocessing)                     │
    │                                                                     │
    │  source_video [B,T,3,480,832]                                       │
    │      │                                                              │
    │      ├──→ VAE.encode → ref_latent [B,F,16,60,104]                  │
    │      │                                                              │
    │  render_video [B,T,3,480,832] (from point cloud rendering)          │
    │      │                                                              │
    │      ├──→ VAE.encode → render_latent [B,F,16,60,104]               │
    │      │                                                              │
    │  mask_video [B,T,3,480,832] (depth masks)                           │
    │      │                                                              │
    │      ├──→ convert_mask_video() → mask_latent [B,F,4,60,104]        │
    │      │    (bilinear downsample to 60×104, group by 4 temporal)     │
    │      │                                                              │
    │  text_prompt                                                        │
    │      │                                                              │
    │      └──→ WanTextEncoder(UM-T5-XXL) → prompt_embeds [B,512,D]      │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     DIFFUSION CORE (per block)                       │
    │                                                                     │
    │  Inputs to CausalWanModel.forward():                                │
    │                                                                     │
    │    noisy_input       [B,C,F,H,W]  (permuted to B,C,F,H,W)          │
    │        +                                                            │
    │    render_block      [B,20,F,60,104] = cat(mask[4], render[16])    │
    │        +                                                            │
    │    prompt_embeds     [B,512,D]                                       │
    │        +                                                            │
    │    timestep          [B,F] or [B]                                    │
    │        +                                                            │
    │    kv_cache          [{k:[B,9360,H,D], v:[B,9360,H,D]} × blocks]  │
    │                                                                     │
    │    ┌─────────────────────────────────────────┐                      │
    │    │  CausalWanModel                         │                      │
    │    │  ├── Patch embedding + RoPE             │                      │
    │    │  ├── N transformer blocks               │                      │
    │    │  │   ├── CausalWanSelfAttention          │                      │
    │    │  │   │   (causal mask + KV cache)        │                      │
    │    │  │   ├── Cross-attention (text context)  │                      │
    │    │  │   └── FFN                             │                      │
    │    │  └── Unpatch + output projection         │                      │
    │    └──────────────────┬──────────────────────┘                      │
    │                       │                                              │
    │                       ▼                                              │
    │    flow_pred [B,F,16,60,104]                                        │
    │    pred_x0 = x_t - sigma_t * flow_pred                              │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     DECODER SIDE (post-processing)                   │
    │                                                                     │
    │  accumulated output [B,F,16,60,104]                                 │
    │      │                                                              │
    │      └──→ VAE.decode → pixel_video [B,T,3,480,832]                 │
    │               │                                                     │
    │               └──→ × 0.5 + 0.5, clamp(0,1) → [0,1] RGB            │
    │                       │                                             │
    │                       └──→ write_video(.mp4, fps=24)               │
    └─────────────────────────────────────────────────────────────────────┘
```

### 1.5 KV Cache State Transitions

```
    KV Cache Lifecycle:
    ═══════════════════

    INIT:    kv_cache[block_i] = {k: zeros[B,9360,H,D], v: zeros[B,9360,H,D]}
                                        │
                                        │  9360 = 1560 × 6 (max context window)
                                        │  1560 = tokens per frame (60 × 104 / 4)
                                        ▼
    BLOCK 0: ┌─────────────────────────────────────────────────┐
             │ Context pass:  ref_block → fills kv[0:4680]     │
             │                (3 frames × 1560 tokens)         │
             │ Denoise pass:  uses kv[0:4680] as context       │
             │                + current block tokens            │
             └─────────────────────────────────────────────────┘
                                        │
                                        ▼
    BLOCK 1: ┌─────────────────────────────────────────────────┐
             │ Context pass:  ref + last_pred → fills more kv  │
             │                kv_size = 4680 + 4680             │
             │ Denoise pass:  attends to full context window    │
             └─────────────────────────────────────────────────┘
                                        │
                                        ▼
    BLOCK N: ┌─────────────────────────────────────────────────┐
             │ Same pattern — sliding window over KV cache     │
             │ Each block sees: ref + last_pred as context     │
             └─────────────────────────────────────────────────┘
```

---

## 2. LTX-2 CastleHill Architecture (for reference)

### 2.1 SCD Inference Flow

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │  LTX-2 SCD (Separable Causal Diffusion)                            │
    │                                                                     │
    │  48-layer DiT → split into:                                         │
    │    ┌──────────────┐     ┌──────────────┐                           │
    │    │ ENCODER       │     │ DECODER       │                           │
    │    │ 32 layers     │     │ 16 layers     │                           │
    │    │ Runs ONCE per │     │ Runs N steps  │                           │
    │    │ frame (σ=0)   │     │ per frame     │                           │
    │    │ + KV cache    │     │ (denoising)   │                           │
    │    └──────┬───────┘     └──────┬───────┘                           │
    │           │                     │                                    │
    │           │  encoder_features   │                                    │
    │           │  (shifted by 1)     │                                    │
    │           └─────────┬───────────┘                                    │
    │                     │                                                │
    │    token_concat: prepend encoder features as prefix to decoder      │
    │    (336 encoder + 336 decoder = 672 tokens/frame)                   │
    └─────────────────────────────────────────────────────────────────────┘
```

### 2.2 VFM v1f Training Flow

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │  VFM v1f — Spherical Cauchy Noise Adapter                           │
    │                                                                     │
    │  text_embeddings [B, T, 4096]                                       │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌──────────────────────────────────┐                               │
    │  │  NoiseAdapterV1b (19.1M params)  │                               │
    │  │  Self-attn + Cross-attn + FFN    │                               │
    │  │  4 layers, 512 hidden            │                               │
    │  │  + Sinusoidal positions (t,h,w)  │                               │
    │  └──────────────┬───────────────────┘                               │
    │                 │ (μ, log_σ) per token [B, 1344, 128]               │
    │                 ▼                                                    │
    │  ┌──────────────────────────────────┐                               │
    │  │  SPHERICAL REINTERPRETATION      │                               │
    │  │  μ̂ = normalize(μ)  → direction   │                               │
    │  │  r = ‖μ‖           → magnitude   │                               │
    │  │  κ = exp(mean(log_σ)) → conc.    │                               │
    │  └──────────────┬───────────────────┘                               │
    │                 │                                                    │
    │                 ▼                                                    │
    │  ┌──────────────────────────────────┐                               │
    │  │  z = r · SphericalCauchy(μ̂, κ)   │                               │
    │  │  (heavy-tailed on S^127)         │                               │
    │  └──────────────┬───────────────────┘                               │
    │                 │ z [B, 1344, 128]                                   │
    │                 ▼                                                    │
    │  ┌──────────────────────────────────┐                               │
    │  │  SigmaHead(μ, x₀)               │                               │
    │  │  Per-token σ_i ∈ [0.05, 0.95]   │                               │
    │  │  Complex → low σ (easy denoise)  │                               │
    │  │  Simple → high σ (can take noise)│                               │
    │  └──────────────┬───────────────────┘                               │
    │                 │                                                    │
    │                 ▼                                                    │
    │  ┌──────────────────────────────────┐                               │
    │  │  48-layer LTX-2 DiT (LoRA)      │                               │
    │  │  1 forward pass @ σ=1            │                               │
    │  │  → velocity v                    │                               │
    │  │  → x̂₀ = z - v                   │                               │
    │  └──────────────────────────────────┘                               │
    │                                                                     │
    │  LOSS = L_mf + L_kl_spherical + L_obs + L_div                      │
    │       + L_mag + L_kappa_pull + L_kappa_entropy                      │
    │       + L_sigma_entropy + L_sigma_pull + L_mu_align                 │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architectural Comparison: InSpatio-World vs LTX-2 CastleHill

```
    ┌─────────────────────────┬────────────────────────────────────────────┐
    │  DIMENSION              │  InSpatio-World       │  LTX-2 CastleHill │
    ├─────────────────────────┼───────────────────────┼────────────────────┤
    │  Base model             │  Wan2.1-I2V 14B       │  LTX-2.3 22B DiT  │
    │  Text encoder           │  UM-T5-XXL            │  Gemma-3 12B       │
    │  VAE latent channels    │  16ch, 60×104         │  128ch, 24×14      │
    │  Frame grouping         │  4 frames/latent      │  1 frame/latent    │
    │  Tokens per frame       │  1560 (60×104/4)      │  336 (24×14)       │
    │  Denoising steps        │  4 (warped from 1000) │  8 or 30 (SCD)     │
    │  Causal mechanism       │  KV cache, block-wise │  SCD enc/dec split │
    │  Depth guidance         │  render + mask latent │  N/A               │
    │  Camera control         │  Trajectory files     │  N/A               │
    │  VFM (1-step)           │  N/A                  │  v1f Spherical     │
    │  Noise structure        │  Standard N(0,I)      │  Spherical Cauchy  │
    │  Flow matching type     │  pred = noise - x₀    │  v = z - x₀        │
    │  Scheduler              │  FlowMatch shifted    │  LTX2Scheduler     │
    └─────────────────────────┴───────────────────────┴────────────────────┘
```

---

## 4. Upgrade Plan: InSpatio-World → LTX-2 v1f

### 4.1 Goal

Replace the Wan2.1-based diffusion backbone with LTX-2's VFM v1f for **1-step video generation**, achieving:
- **8x faster inference** (1 DiT pass instead of 4 denoising steps)
- **Structured noise** via Spherical Cauchy adapter (text-conditioned)
- Maintain depth-guided novel-view synthesis capabilities

### 4.2 Architecture Target

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │          InSpatio-World + LTX-2 VFM v1f (TARGET)                    │
    │                                                                     │
    │  source_video + depth + trajectory                                  │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐      │
    │  │ LTX-2.3 VAE    │   │ Gemma-3 12B    │   │ DA3 Depth      │      │
    │  │ Encode video    │   │ Text encode    │   │ Estimation     │      │
    │  │ → [B,F,128,    │   │ → [B,T,4096]   │   │ → render+mask  │      │
    │  │    24,14]       │   │                │   │                │      │
    │  └────────┬───────┘   └────────┬───────┘   └────────┬───────┘      │
    │           │                    │                     │               │
    │           ▼                    ▼                     ▼               │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │  NoiseAdapterV1b (19.1M params)                         │        │
    │  │  Inputs: text_embeddings + depth_positions              │        │
    │  │  Spherical Cauchy: z = ‖μ‖ · SpCauchy(μ̂, κ)            │        │
    │  │  Per-token σ via SigmaHead(μ, x₀)                      │        │
    │  └──────────────────────┬──────────────────────────────────┘        │
    │                         │ z [B, 1344, 128]                          │
    │                         ▼                                           │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │  LTX-2.3 48-layer DiT (LoRA, 1 forward pass)           │        │
    │  │  + Depth-conditioned (render_latent via concat or       │        │
    │  │    cross-attention)                                      │        │
    │  │  → velocity v → x̂₀ = z - v                             │        │
    │  └──────────────────────┬──────────────────────────────────┘        │
    │                         │                                           │
    │                         ▼                                           │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │  LTX-2.3 VAE Decoder                                    │        │
    │  │  Latent → pixel video [B, T, 3, 1024, 1024]            │        │
    │  └─────────────────────────────────────────────────────────┘        │
    └─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation Phases

#### Phase 1: Model Swap Foundation (Week 1-2)

**Goal:** Replace Wan2.1 backbone with LTX-2.3, keeping multi-step denoising initially.

```
    CURRENT                           TARGET (Phase 1)
    ═══════                           ═════════════════
    WanDiffusionWrapper               LTXDiffusionWrapper
    ├── CausalWanModel (14B)          ├── LTXModel (22B)
    ├── FlowMatchScheduler            ├── LTX2Scheduler
    └── UM-T5-XXL text encoder        └── Gemma-3 12B text encoder

    WanVAEWrapper                     LTXVAEWrapper
    ├── 16ch latent                   ├── 128ch latent
    ├── 60×104 spatial                ├── varies by resolution
    └── 4-frame temporal group        └── 1-frame temporal

    KEEP UNCHANGED:
    ├── DA3 depth estimation pipeline
    ├── Point cloud rendering
    ├── Trajectory control system
    └── Florence-2 captioning (or replace with Gemma-3)
```

**Files to create/modify:**

| Action | File | Description |
|--------|------|-------------|
| **NEW** | `utils/ltx_wrapper.py` | LTX-2.3 model wrappers (replaces `wan_wrapper.py`) |
| **NEW** | `utils/ltx_scheduler.py` | LTX-2 sigma scheduler (replaces `scheduler.py`) |
| **MODIFY** | `pipeline/causal_inference.py` | Update to use LTX-2 model API |
| **MODIFY** | `inference_causal_test.py` | Update model loading and config |
| **MODIFY** | `configs/inference.yaml` | LTX-2 model paths and parameters |
| **NEW** | `configs/inference_ltx2.yaml` | LTX-2 specific config |

**Key challenges:**
1. **Latent space mismatch**: Wan uses 16ch, LTX uses 128ch. The `convert_mask_video()` and all latent tensor shapes change.
2. **Token count**: Wan has 1560 tokens/frame, LTX has 336. KV cache dimensions shrink 4.6x.
3. **Depth injection**: Wan concatenates render+mask into input channels (`in_dim=36`). LTX needs a new conditioning mechanism.

#### Phase 2: Depth Conditioning for LTX-2 (Week 2-3)

**Goal:** Adapt depth/render guidance to work with LTX-2's architecture.

**Options (ranked by feasibility):**

1. **Channel concatenation (simplest, recommended first)**
   - Encode render+mask through LTX VAE → depth_latent [B,F,128,H,W]
   - Concatenate with noisy_latent along channel dim → [B,F,256,H,W]
   - Add a learned projection layer (256→128) before DiT
   - Requires LoRA training on this expanded input

2. **Cross-attention injection**
   - Encode depth as separate token sequence
   - Add cross-attention layers that attend to depth tokens
   - More elegant but requires modifying DiT architecture

3. **ControlNet-style adapter**
   - Train a lightweight depth encoder that injects features into DiT layers
   - Most flexible, doesn't modify base model
   - Higher training cost

**Recommended: Option 1** — channel concat is how InSpatio-World already works with Wan, and it minimizes architectural changes.

#### Phase 3: VFM v1f Integration (Week 3-5)

**Goal:** Replace multi-step denoising with 1-step VFM.

```
    BEFORE (Phase 1):                 AFTER (Phase 3):
    ═══════════════════               ═══════════════════
    noise ~ N(0,I)                    z ~ NoiseAdapterV1b(text, depth_pos)
    4 denoising steps                 1 forward pass
    re-noise between steps            z = ‖μ‖ · SpCauchy(μ̂, κ)
    ~4s per block                     ~0.5s per block
```

**Key integration steps:**

1. **Port NoiseAdapterV1b** from CastleHill
   - Copy `packages/ltx-core/src/ltx_core/models/noise_adapter_v1b.py`
   - Copy `packages/ltx-trainer/src/ltx_trainer/spherical_utils.py`
   - Adapt position encoding for InSpatio resolution

2. **Port SigmaHead** from CastleHill
   - Copy sigma head MLP (128→256→256→1)
   - Needed for per-token denoising schedule

3. **Modify inference pipeline**
   ```python
   # OLD: Multi-step denoising
   for t in [1000, 750, 500, 250]:
       flow_pred, x0 = model(noisy, t, ...)
       noisy = scheduler.add_noise(x0, noise, next_t)

   # NEW: VFM 1-step
   z = noise_adapter(text_embeds, depth_positions)  # Structured noise
   sigma = sigma_head(z, ref_latent)                # Per-token schedule
   v_pred = model(z, sigma, ...)                    # 1 forward pass
   x0_pred = z - v_pred                             # Clean video
   ```

4. **Train LoRA + adapter on depth-conditioned data**
   - Use InSpatio's paired (source, target, depth) training data
   - Add depth positions to NoiseAdapterV1b's position encoding
   - Loss = v1f losses + depth reconstruction loss

#### Phase 4: SCD for Long-Form Streaming (Week 5-7)

**Goal:** Enable autoregressive long-form video via SCD encoder/decoder split.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │  LTX-2 SCD + VFM v1f + InSpatio Depth                          │
    │                                                                  │
    │  For each frame t:                                               │
    │                                                                  │
    │  1. ENCODER (32 layers, runs once):                              │
    │     Input: clean frames [0..t-1] + depth context                │
    │     KV-cache: O(1) per new frame                                │
    │     Output: encoder_features (cached)                            │
    │                                                                  │
    │  2. NOISE ADAPTER (VFM v1f):                                     │
    │     z_t = NoiseAdapter(text, depth_positions_t)                  │
    │     σ_t = SigmaHead(z_t, ref_latent_t)                          │
    │                                                                  │
    │  3. DECODER (16 layers, 1 step via VFM):                         │
    │     Input: z_t + token_concat(encoder_features)                  │
    │     Output: v_pred → x̂₀_t = z_t - v_pred                       │
    │                                                                  │
    │  4. ACCUMULATE: append x̂₀_t to output buffer                    │
    │                                                                  │
    │  Per-frame cost: 1 encoder pass (32 layers) + 1 decoder pass    │
    │                  (16 layers) = 48 layers total                    │
    │  vs current: 4 steps × 14B model = ~56B FLOPs per block         │
    └──────────────────────────────────────────────────────────────────┘
```

### 4.4 Training Plan for v1f Upgrade

```
    STAGE 1: Base LTX-2 → InSpatio (depth-conditioned)
    ═══════════════════════════════════════════════════
    Data:    InSpatio paired videos (source + target + depth + camera)
    Model:   LTX-2.3 base + LoRA (r=32) + depth input projection
    Loss:    Standard flow matching (multi-step, 8 steps)
    Steps:   ~5K on 5000 video pairs
    Goal:    LTX-2 learns depth-guided novel-view synthesis

    STAGE 2: VFM v1f adapter training
    ═══════════════════════════════════
    Data:    Same paired data + precomputed trajectories
    Model:   Frozen LTX-2 (from Stage 1) + NoiseAdapterV1b + SigmaHead
    Loss:    v1f loss suite (L_mf + L_kl_spherical + L_obs + ...)
    Steps:   ~10K
    Goal:    1-step generation with depth conditioning

    STAGE 3: SCD split + VFM (optional, for streaming)
    ════════════════════════════════════════════════════
    Data:    Longer video sequences (30+ frames)
    Model:   SCD(LTX-2) = 32-encoder + 16-decoder + VFM adapter
    Loss:    SCD training strategy + v1f losses
    Steps:   ~2K
    Goal:    Autoregressive streaming with 1-step per frame
```

### 4.5 File Mapping: What to Port from CastleHill

```
    CastleHill                              →  InSpatio-World
    ══════════                                 ═══════════════
    packages/ltx-core/
    ├── scd_model.py                        →  wan/modules/scd_model.py
    ├── models/noise_adapter_v1b.py         →  utils/noise_adapter.py
    └── components/schedulers.py            →  utils/ltx_scheduler.py

    packages/ltx-trainer/
    ├── spherical_utils.py                  →  utils/spherical_utils.py
    ├── training_strategies/
    │   └── vfm_strategy_v1f.py             →  training/vfm_v1f_strategy.py
    ├── configs/
    │   └── ltx2_vfm_v1f_spherical.yaml     →  configs/train_vfm_v1f.yaml
    └── scripts/
        └── scd_inference.py                →  pipeline/scd_causal_inference.py

    KEEP from InSpatio:
    ├── depth/                              (DA3 depth estimation - unchanged)
    ├── datasets/                           (data loading - update latent shapes)
    ├── scripts/gen_json.py                 (captioning - maybe replace w/ Gemma-3)
    ├── scripts/render_point_cloud.py       (3D rendering - unchanged)
    ├── traj/                               (camera trajectories - unchanged)
    └── utils/render_warper.py              (adapt for LTX latent dims)
```

### 4.6 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| LTX-2 22B too large for consumer GPU | High | int8-quanto quantization (11GB), split across GPUs |
| Depth conditioning quality drops | Medium | Start with channel concat (proven in Wan), A/B test |
| VFM v1f not validated with depth | High | Train Stage 1 (multi-step) first, verify quality before VFM |
| KV cache memory with larger model | Medium | LTX tokens/frame (336) < Wan (1560), actually helps |
| Training data format change | Low | Write converter script for latent re-encoding |
| Gemma-3 vs UM-T5 text mismatch | Low | Use Gemma-3 (LTX native), retrain adapter |

### 4.7 Expected Performance

```
    ┌────────────────────────────┬──────────────┬──────────────┐
    │  Metric                    │  Current     │  After v1f   │
    │                            │  (Wan2.1)    │  (LTX-2)     │
    ├────────────────────────────┼──────────────┼──────────────┤
    │  Denoising steps/block     │  4           │  1 (VFM)     │
    │  Time per 3-frame block    │  ~3-4s       │  ~0.5-0.8s   │
    │  30-frame video            │  ~40s        │  ~5-8s       │
    │  Max resolution            │  480p        │  1024p        │
    │  Max duration              │  Limited     │  30s+ (SCD)  │
    │  Model size                │  14B         │  22B (11B q8)│
    │  VRAM (inference)          │  ~24GB       │  ~16GB (q8)  │
    │  Text encoder              │  UM-T5-XXL   │  Gemma-3 12B │
    └────────────────────────────┴──────────────┴──────────────┘
```

---

## 5. Quick Start: Phase 1 Implementation

### Step 1: Create LTX wrapper

```python
# utils/ltx_wrapper.py (skeleton)
class LTXDiffusionWrapper(nn.Module):
    """Drop-in replacement for WanDiffusionWrapper using LTX-2.3."""
    def __init__(self, model_path, quantization="int8-quanto", ...):
        self.model = LTXModel.from_pretrained(model_path)
        # Apply quantization
        # Apply LoRA
        # Add depth input projection

    def forward(self, noisy_latent, timestep, context,
                kv_cache=None, render_latent=None, ...):
        # Concat depth if provided
        # Forward through LTX DiT
        # Return flow_pred, pred_x0
```

### Step 2: Adapt CausalInferencePipeline

```python
# pipeline/causal_inference_ltx.py
class LTXCausalInferencePipeline(nn.Module):
    def __init__(self, args, device):
        self.generator = LTXDiffusionWrapper(...)
        self.text_encoder = GemmaTextEncoder(...)  # or keep UM-T5
        self.vae = LTXVAEWrapper(...)
        # Smaller KV cache: 336 tokens/frame vs 1560
        self.frame_seq_length = 336

    def inference_vfm(self, text_prompts, ref_latent, ...):
        """1-step VFM inference path."""
        # Noise adapter → structured noise
        z = self.noise_adapter(text_embeds, depth_positions)
        # Single forward pass
        v_pred = self.generator(z, sigma=1.0, context=text_embeds, ...)
        x0_pred = z - v_pred
        # Decode
        return self.vae.decode(x0_pred)
```

---

## 6. Phase 1 Implementation — Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `utils/spherical_utils.py` | ~110 | Spherical Cauchy sampling, KL, SLERP, geodesic distance |
| `utils/noise_adapter.py` | ~260 | NoiseAdapterV1b (19.1M) + SigmaHead (per-token sigma) |
| `utils/ltx_scheduler.py` | ~130 | LTX2Scheduler + LTXFlowMatchScheduler bridge |
| `utils/ltx_wrapper.py` | ~350 | LTXDiffusionWrapper, LTXTextEncoder, LTXVAEWrapper + LoRA |
| `pipeline/causal_inference_ltx.py` | ~320 | LTXCausalInferencePipeline + denoise_block_ltx + denoise_block_vfm |
| `configs/inference_ltx2.yaml` | ~70 | LTX-2 config (model paths, quantization, VFM toggles) |
| `inference_ltx2_test.py` | ~170 | Entry point script (parallel to inference_causal_test.py) |

### Dependency Chain

```
inference_ltx2_test.py
  └─ pipeline/causal_inference_ltx.py
       ├─ utils/ltx_wrapper.py
       │    ├─ utils/ltx_scheduler.py
       │    └─ ltx_trainer.model_loader (from CastleHill packages)
       ├─ utils/noise_adapter.py (VFM mode only)
       └─ utils/spherical_utils.py (VFM mode only)
```

### What's Needed to Run

1. **Install ltx-core + ltx-trainer** from CastleHill as packages:
   ```bash
   cd /path/to/ltx2-castlehill
   pip install -e packages/ltx-core
   pip install -e packages/ltx-trainer
   ```

2. **Download LTX-2.3 model** (~11GB int8-quanto):
   ```bash
   # Place at ./checkpoints/ltx-2.3-22b-dev.safetensors
   # Or configure path in configs/inference_ltx2.yaml
   ```

3. **Gemma-3 text encoder** (optional — can use cached embeddings):
   ```bash
   # Place at ./checkpoints/gemma/
   ```
