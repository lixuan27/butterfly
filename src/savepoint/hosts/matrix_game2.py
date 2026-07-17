"""Matrix-Game 2.0 host adapter: a stepwise, saveable driver for MG2 streaming.

MG2's own ``inference_streaming.py`` runs a terminal-interactive loop with all
world state in Python locals. This adapter re-owns that loop behind the
StreamingHost interface so the state can be captured/restored at any block
boundary.

Runtime-state inventory this adapter serializes (docs/WORLDSTATE_SPEC.md):
  caches:  kv (30), kv_mouse (30), kv_keyboard (30), cross (30)
  tensors: vae_cache.00..32 (streaming VAE conv tails), latents_window (last
           `local_attn_size` clean latent frames), img_latent0 (first-frame
           VAE latent), visual_context (CLIP), first-frame RGB (for reprime)
  cursor:  latent_frame; actions: per-block {keys,mouse}; rng: torch cpu+cuda

Requires the MG2 source tree on PYTHONPATH (third_party/mg2_src_fetch/Matrix-Game-2)
and the Skywork/Matrix-Game-2.0 weights on disk. GPU only.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch
from einops import rearrange
from omegaconf import OmegaConf

from ..state import WorldState
from .base import StepResult, StreamingHost

# action schemas per MG2 mode: name -> one-hot/analog encoders
KEYBOARD_MAP_UNIVERSAL = {"w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0],
                          "d": [0, 0, 0, 1], "": [0, 0, 0, 0]}
MOUSE_SCALE = 0.1


class MatrixGame2Host(StreamingHost):
    """Streaming driver for the universal-scene distilled Matrix-Game 2.0."""

    def __init__(
        self,
        mg2_root: str,
        ckpt_dir: str,
        *,
        mode: str = "universal",
        max_latent_frames: int = 360,
        device: str = "cuda",
        compile_vae: bool = True,
        latents_window_frames: int = 0,
    ):
        """latents_window_frames: how many recent clean latent frames the host
        retains (>= local_attn_size). Longer windows enable higher-fidelity
        reprime saves (the M3 rate-distortion dial); 0 = model's attention window."""
        if mode != "universal":
            raise NotImplementedError("P0 targets the universal model; gta/templerun later")
        self.mode = mode
        self.device = torch.device(device)
        self.weight_dtype = torch.bfloat16
        self.max_latent_frames = max_latent_frames
        self.model_id = "matrix-game-2.0/base_distill"

        # ---- build models exactly as inference_streaming.py does ----
        from pipeline import CausalInferenceStreamingPipeline
        from demo_utils.vae_block3 import VAEDecoderWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper
        from utils.wan_wrapper import WanDiffusionWrapper
        from safetensors.torch import load_file

        config = OmegaConf.load(os.path.join(
            mg2_root, "configs/inference_yaml", f"inference_{mode}.yaml"))
        # MG2 yamls carry model_config as a path relative to the MG2 repo root;
        # absolutize so the host works from any CWD (first smoke died on this)
        mk = config.get("model_kwargs", None)
        if mk is not None and "model_config" in mk and not os.path.isabs(mk["model_config"]):
            mk["model_config"] = os.path.join(mg2_root, mk["model_config"])
        self.config = config

        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}), is_causal=True)

        vae_decoder = VAEDecoderWrapper()
        vae_sd = torch.load(os.path.join(ckpt_dir, "Wan2.1_VAE.pth"), map_location="cpu")
        dec_sd = {k: v for k, v in vae_sd.items() if "decoder." in k or "conv2" in k}
        vae_decoder.load_state_dict(dec_sd)
        vae_decoder.to(self.device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        if compile_vae:
            vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        self.pipeline = CausalInferenceStreamingPipeline(
            config, generator=generator, vae_decoder=vae_decoder)
        sd = load_file(os.path.join(ckpt_dir, "base_distilled_model/base_distill.safetensors"))
        self.pipeline.generator.load_state_dict(sd)
        self.pipeline = self.pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(ckpt_dir, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

        from demo_utils.constant import ZERO_VAE_CACHE
        self._zero_vae_cache_len = len(ZERO_VAE_CACHE)

        p = self.pipeline
        self.num_frame_per_block = p.num_frame_per_block          # 3
        self.frame_seq_length = p.frame_seq_length                # 880
        self.local_attn_size = p.local_attn_size                  # 6
        self.kb_dim = 4
        self.window_len = max(self.local_attn_size, latents_window_frames)

        # ---- host-owned episode state (populated by prime/restore) ----
        self._episode_active = False
        self._latent_frame = 0
        self._actions: List[Dict[str, Any]] = []
        self._vae_cache: List[Optional[torch.Tensor]] = []
        self._latents_window: Optional[torch.Tensor] = None  # [1,16,<=local_attn,44,80]
        self._cond: Optional[Dict[str, torch.Tensor]] = None
        self._img_latent0: Optional[torch.Tensor] = None
        self._first_frame: Optional[torch.Tensor] = None     # [1,3,1,352,640] in [-1,1]
        # dedicated noise stream: immune to global-RNG pollution (torch.compile
        # autotune benchmarks consume the global CUDA RNG) and the natural
        # handle for fork semantics
        self._rng = torch.Generator(device=self.device)

    # ------------------------------------------------------------------ prime

    @torch.no_grad()
    def _build_episode_cond(self, img: torch.Tensor) -> torch.Tensor:
        """VAE-encode [first frame, gray padding] into the episode-length
        cond_concat. Deterministic (encode returns mu, no sampling), so prime()
        and restore() reproduce bit-identical conditioning from the same frame.
        NOTE: padding-frame latents are NOT zero — the VAE normalizes with
        nonzero dataset statistics — which is why restore must re-encode
        rather than zero-fill (found by verify #2)."""
        padding = torch.zeros_like(img).repeat(1, 1, 4 * (self.max_latent_frames - 1), 1, 1)
        tiler = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(torch.cat([img, padding], dim=2),
                                   device=self.device, **tiler).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        return torch.cat([mask_cond[:, :4], img_cond], dim=1)

    @torch.no_grad()
    def prime(self, image: Any, *, seed: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """image: preprocessed tensor [1,3,1,352,640] in [-1,1] (host caller
        handles file loading/resize so the adapter stays PIL-free)."""
        self._rng.manual_seed(seed)
        img = image.to(dtype=self.weight_dtype, device=self.device)
        self._first_frame = img.detach().clone()

        cond_concat = self._build_episode_cond(img)
        visual_context = self.vae.clip.encode_video(img)

        n_pix = 1 + 4 * (self.max_latent_frames - 1)
        self._cond = {
            "cond_concat": cond_concat.to(self.weight_dtype),
            "visual_context": visual_context.to(self.weight_dtype),
            "keyboard_cond": torch.zeros(1, n_pix, self.kb_dim,
                                         dtype=self.weight_dtype, device=self.device),
            "mouse_cond": torch.zeros(1, n_pix, 2,
                                      dtype=self.weight_dtype, device=self.device),
        }
        self._img_latent0 = cond_concat[:, 4:, :1].detach().clone()

        self._init_caches()
        self._vae_cache = [None] * self._zero_vae_cache_len
        self._latent_frame = 0
        self._actions = []
        self._latents_window = None
        self._episode_active = True

    def _init_caches(self) -> None:
        p = self.pipeline
        p.kv_cache1 = p.kv_cache_mouse = p.kv_cache_keyboard = p.crossattn_cache = None
        p._initialize_kv_cache(batch_size=1, dtype=self.weight_dtype, device=self.device)
        p._initialize_kv_cache_mouse_and_keyboard(batch_size=1, dtype=self.weight_dtype,
                                                  device=self.device)
        p._initialize_crossattn_cache(batch_size=1, dtype=self.weight_dtype,
                                      device=self.device)

    # ------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, action: Dict[str, Any], *, n_latent_frames: int = 0) -> StepResult:
        """Advance one block (num_frame_per_block latent frames) under `action`
        = {"keys": ["w"], "mouse": [dpitch, dyaw]}. n_latent_frames is fixed to
        the model's block size; a nonzero override raises."""
        if not self._episode_active:
            raise RuntimeError("prime() or restore() first")
        if n_latent_frames not in (0, self.num_frame_per_block):
            raise ValueError(f"MG2 steps in blocks of {self.num_frame_per_block}")
        nfpb = self.num_frame_per_block
        csf = self._latent_frame
        if csf + nfpb > self.max_latent_frames:
            raise RuntimeError("episode exhausted (RoPE table budget); start a new one")
        p = self.pipeline
        t0 = time.time()

        # --- write the action into the pixel-frame range of this block (their
        # cond_current(replace=...) semantics, causal_inference.py:110-132) ---
        keys = action.get("keys", [])
        kb_vec = torch.tensor(
            [sum(v) for v in zip(*[KEYBOARD_MAP_UNIVERSAL[k] for k in keys])]
            if keys else KEYBOARD_MAP_UNIVERSAL[""],
            dtype=self.weight_dtype, device=self.device).clamp(0, 1)
        ms_vec = torch.tensor(action.get("mouse", [0.0, 0.0]),
                              dtype=self.weight_dtype, device=self.device)
        final_frame = 1 + 4 * (csf + nfpb - 1)
        last_n = 1 + 4 * (nfpb - 1) if csf == 0 else 4 * nfpb
        self._cond["keyboard_cond"][:, final_frame - last_n: final_frame] = kb_vec
        self._cond["mouse_cond"][:, final_frame - last_n: final_frame] = ms_vec

        new_cond = {
            "cond_concat": self._cond["cond_concat"][:, :, csf: csf + nfpb],
            "visual_context": self._cond["visual_context"],
            "keyboard_cond": self._cond["keyboard_cond"][:, :final_frame],
            "mouse_cond": self._cond["mouse_cond"][:, :final_frame],
        }

        # --- per-block noise from the host's dedicated generator (fork-friendly:
        # its state at a block boundary determines the whole remaining stream) ---
        noisy_input = torch.randn(1, 16, nfpb, 44, 80, generator=self._rng,
                                  device=self.device, dtype=self.weight_dtype)

        # --- 3-step distilled denoise (pipeline.inference inner loop) ---
        for index, current_timestep in enumerate(p.denoising_step_list):
            timestep = torch.ones([1, nfpb], device=self.device,
                                  dtype=torch.int64) * current_timestep
            _, denoised_pred = p.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=new_cond,
                timestep=timestep,
                kv_cache=p.kv_cache1,
                kv_cache_mouse=p.kv_cache_mouse,
                kv_cache_keyboard=p.kv_cache_keyboard,
                crossattn_cache=p.crossattn_cache,
                current_start=csf * self.frame_seq_length,
            )
            if index < len(p.denoising_step_list) - 1:
                next_t = p.denoising_step_list[index + 1]
                flat = rearrange(denoised_pred, "b c f h w -> (b f) c h w")
                step_noise = torch.randn(flat.shape, generator=self._rng,
                                         device=flat.device, dtype=flat.dtype)
                noisy_input = rearrange(
                    p.scheduler.add_noise(
                        flat, step_noise,
                        next_t * torch.ones([nfpb], device=self.device, dtype=torch.long)),
                    "(b f) c h w -> b c f h w", b=1)

        # --- context rerun at context_noise (=0) to roll the clean KV cache ---
        context_timestep = torch.ones([1, nfpb], device=self.device,
                                      dtype=torch.int64) * self.config.context_noise
        p.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=new_cond,
            timestep=context_timestep,
            kv_cache=p.kv_cache1,
            kv_cache_mouse=p.kv_cache_mouse,
            kv_cache_keyboard=p.kv_cache_keyboard,
            crossattn_cache=p.crossattn_cache,
            current_start=csf * self.frame_seq_length,
        )

        # --- streaming VAE decode with cache handoff ---
        latents_btchw = denoised_pred.transpose(1, 2)  # [1, f, 16, 44, 80]
        video, vae_cache = p.vae_decoder(latents_btchw.half(), *self._vae_cache)
        self._vae_cache = list(vae_cache)

        # --- host bookkeeping ---
        self._latent_frame = csf + nfpb
        self._actions.append({"latent_frame": csf, "keys": list(keys),
                              "mouse": [float(x) for x in action.get("mouse", [0, 0])]})
        window = self._latents_window
        joined = denoised_pred if window is None else torch.cat([window, denoised_pred], dim=2)
        self._latents_window = joined[:, :, -self.window_len:].detach().clone()

        frames = (video.float() * 0.5 + 0.5).clamp(0, 1)[0]  # [T,C,H,W] in [0,1]
        dt_ms = (time.time() - t0) * 1000
        return StepResult(
            frames=frames,
            latents=denoised_pred[0].detach(),
            frame_index=1 + 4 * (csf - 1) if csf > 0 else 0,
            stats={"block_ms": dt_ms, "fps": frames.shape[0] / (dt_ms / 1000)},
        )

    # ---------------------------------------------------------------- capture

    def capture(self, *, mode: str = "state", reprime_window: int = 0) -> WorldState:
        """reprime_window: for mode='reprime', store only the last n latent
        frames (must be a multiple of the block size; 0 = all retained)."""
        p = self.pipeline
        window = self._latents_window
        if reprime_window:
            if reprime_window % self.num_frame_per_block or reprime_window > window.shape[2]:
                raise ValueError(f"bad reprime_window {reprime_window} "
                                 f"(retained {window.shape[2]}, block {self.num_frame_per_block})")
            window = window[:, :, -reprime_window:]
        tensors: Dict[str, torch.Tensor] = {
            "latents_window": window,
            "img_latent0": self._img_latent0,
            "visual_context": self._cond["visual_context"],
            "first_frame": self._first_frame,
            "rng_gen_state": self._rng.get_state(),  # host noise stream (uint8)
        }
        caches = None
        if mode == "state":
            caches = {
                "kv": p.kv_cache1,
                "kv_mouse": p.kv_cache_mouse,
                "kv_keyboard": p.kv_cache_keyboard,
                "cross": p.crossattn_cache,
            }
            for i, t in enumerate(self._vae_cache):
                if t is not None:
                    tensors[f"vae_cache.{i:02d}"] = t
        meta = {
            "model_id": self.model_id,
            "mode": self.mode,
            "resolution": [352, 640],
            "num_frame_per_block": self.num_frame_per_block,
            "local_attn_size": self.local_attn_size,
            "max_latent_frames": self.max_latent_frames,
            "gpu": torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else "cpu",
        }
        return WorldState.capture(
            caches, tensors, mode=mode,
            cursor={"latent_frame": self._latent_frame,
                    "pixel_frame": max(0, 1 + 4 * (self._latent_frame - 1))},
            actions=self._actions, meta=meta, device=self.device)

    # ---------------------------------------------------------------- restore

    @torch.no_grad()
    def restore(self, state: WorldState) -> None:
        if state.meta.get("model_id") != self.model_id:
            raise ValueError(f"save is for {state.meta.get('model_id')}, host is {self.model_id}")
        self._restore_common(state)
        if state.mode == "state":
            self._restore_full(state)
        else:
            self._reprime(state)
        self._episode_active = True

    def _restore_common(self, state: WorldState) -> None:
        dev, dt = self.device, self.weight_dtype
        self._latent_frame = int(state.cursor["latent_frame"])
        self._actions = [dict(a) for a in state.actions]
        self._latents_window = state.tensors["latents_window"].to(dev, dt)
        self._img_latent0 = state.tensors["img_latent0"].to(dev, dt)
        self._first_frame = state.tensors["first_frame"].to(dev, dt)
        self._rng.set_state(state.tensors["rng_gen_state"].to("cpu", torch.uint8))

        # Rebuild episode conditioning by re-encoding the saved first frame
        # through the same deterministic path prime() uses. Zero-filling the
        # padding frames is WRONG: their VAE latents are nonzero (normalized
        # with nonzero dataset statistics) and every step reads its slice.
        cond_concat = self._build_episode_cond(self._first_frame)
        n_pix = 1 + 4 * (self.max_latent_frames - 1)
        self._cond = {
            "cond_concat": cond_concat.to(dt),
            "visual_context": state.tensors["visual_context"].to(dev, dt),
            "keyboard_cond": torch.zeros(1, n_pix, self.kb_dim, dtype=dt, device=dev),
            "mouse_cond": torch.zeros(1, n_pix, 2, dtype=dt, device=dev),
        }
        nfpb = self.num_frame_per_block
        for a in self._actions:
            csf = int(a["latent_frame"])
            kb = torch.tensor(
                [sum(v) for v in zip(*[KEYBOARD_MAP_UNIVERSAL[k] for k in a["keys"]])]
                if a["keys"] else KEYBOARD_MAP_UNIVERSAL[""], dtype=dt, device=dev).clamp(0, 1)
            ms = torch.tensor(a["mouse"], dtype=dt, device=dev)
            final = 1 + 4 * (csf + nfpb - 1)
            last_n = 1 + 4 * (nfpb - 1) if csf == 0 else 4 * nfpb
            self._cond["keyboard_cond"][:, final - last_n: final] = kb
            self._cond["mouse_cond"][:, final - last_n: final] = ms

    def _restore_full(self, state: WorldState) -> None:
        p = self.pipeline
        if p.kv_cache1 is None:
            self._init_caches()
        state.restore_caches(
            {"kv": p.kv_cache1, "kv_mouse": p.kv_cache_mouse,
             "kv_keyboard": p.kv_cache_keyboard, "cross": p.crossattn_cache},
            device=self.device)
        self._vae_cache = [None] * self._zero_vae_cache_len
        for name, t in state.tensors.items():
            if name.startswith("vae_cache."):
                self._vae_cache[int(name.split(".")[1])] = t.to(self.device, torch.float16)

    def _reprime(self, state: WorldState) -> None:
        """Rebuild all caches from the saved latent window via the model's own
        timestep-0 context pass at the correct absolute positions (RoPE is
        global), then re-warm the VAE cache by decoding the window."""
        from ..state import restore_rng
        p = self.pipeline
        self._init_caches()
        nfpb = self.num_frame_per_block
        window = self._latents_window
        n_win = window.shape[2]
        if n_win % nfpb != 0:
            raise ValueError(f"latent window {n_win} not a multiple of block {nfpb}")
        start = self._latent_frame - n_win
        if start < 0:
            raise ValueError("cursor/window mismatch")

        # The upstream initial_latent path only supports priming from frame 0
        # (cache global_end grows from 0). For a mid-episode reprime at absolute
        # positions, pre-seat every cache group's global cursor at the window
        # start so the first write appends cleanly (retry1 crashed here: fresh
        # global_end=0 vs current_start=26400 made the roll arithmetic negative).
        start_tok = start * self.frame_seq_length
        for blk in p.kv_cache1:
            blk["global_end_index"].fill_(start_tok)
            blk["local_end_index"].fill_(0)
        for group in (p.kv_cache_mouse, p.kv_cache_keyboard):
            for blk in group:  # action caches count in latent frames, not tokens
                blk["global_end_index"].fill_(start)
                blk["local_end_index"].fill_(0)

        self._vae_cache = [None] * self._zero_vae_cache_len
        timestep = torch.zeros([1, nfpb], device=self.device, dtype=torch.int64)
        for b in range(n_win // nfpb):
            csf = start + b * nfpb
            block = window[:, :, b * nfpb:(b + 1) * nfpb]
            final = 1 + 4 * (csf + nfpb - 1)
            new_cond = {
                "cond_concat": self._cond["cond_concat"][:, :, csf: csf + nfpb],
                "visual_context": self._cond["visual_context"],
                "keyboard_cond": self._cond["keyboard_cond"][:, :final],
                "mouse_cond": self._cond["mouse_cond"][:, :final],
            }
            p.generator(
                noisy_image_or_video=block,
                conditional_dict=new_cond,
                timestep=timestep,
                kv_cache=p.kv_cache1,
                kv_cache_mouse=p.kv_cache_mouse,
                kv_cache_keyboard=p.kv_cache_keyboard,
                crossattn_cache=p.crossattn_cache,
                current_start=csf * self.frame_seq_length,
            )
            _, vae_cache = p.vae_decoder(block.transpose(1, 2).half(), *self._vae_cache)
            self._vae_cache = list(vae_cache)
        if state.rng:
            restore_rng(state.rng, self.device)

    # ------------------------------------------------------------- properties

    @property
    def cursor(self) -> Dict[str, int]:
        return {"latent_frame": self._latent_frame,
                "pixel_frame": max(0, 1 + 4 * (self._latent_frame - 1))}

    @property
    def action_history(self) -> List[Dict[str, Any]]:
        return list(self._actions)

    def state_components(self) -> Dict[str, int]:
        return self.capture(mode="state").nbytes()
