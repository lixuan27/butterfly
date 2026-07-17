"""WorldState: the complete runtime state of a streaming world model.

A streaming, KV-cached causal video diffusion model (Self-Forcing family:
Matrix-Game 2.0, Self-Forcing, CausVid, ...) keeps its entire "game state" in
a handful of runtime buffers. For Matrix-Game 2.0 the full list is:

  * ``kv``           — 30 blocks of self-attention K/V over latent tokens
  * ``kv_mouse``     — 30 blocks of per-spatial-token temporal K/V over mouse actions
  * ``kv_keyboard``  — 30 blocks of temporal K/V over keyboard actions
  * ``cross``        — 30 blocks of conditioning K/V (CLIP visual context; static)
  * the streaming VAE decoder's causal conv cache (host-owned, passed by hand)
  * the rolling clean-latent context, frame cursor, action history
  * the sampler RNG state (``torch.randn_like`` inside the denoise loop)

Nothing else persists between frames. Which *subsets* of this suffice to
resume a world is an empirical question (SaveBench M1/M3); both save modes
below exist precisely to measure it:

  * mode="state"   — full runtime state (upper bound: bit-exact resume)
  * mode="reprime" — only recent clean latents + cursor + RNG; restore replays
                     the model's own context-caching pass (MB instead of GB;
                     fidelity is measured, not assumed)

WorldState is deliberately model-agnostic: it stores *named groups* of block
caches plus *named* flat tensors. Each host adapter decides what goes in.

The on-disk ``.wsave`` container is a plain ZIP holding a JSON header, one
safetensors blob for all tensors, and JSON action/RNG sidecars. No pickle
anywhere: a save file received from a stranger must be safe to open.
"""

from __future__ import annotations

import base64
import json
import random
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

try:
    from safetensors.torch import load as st_load
    from safetensors.torch import save as st_save
except ImportError as e:  # pragma: no cover
    raise ImportError("savepoint requires `safetensors` (pip install safetensors)") from e

WSAVE_VERSION = 0
_HEADER_NAME = "header.json"
_TENSORS_NAME = "tensors.safetensors"
_ACTIONS_NAME = "actions.json"
_RNG_NAME = "rng.json"

#: cache-entry keys that hold the big quantizable payloads (vs exact cursors)
_PAYLOAD_KEYS = ("k", "v")

BlockCache = List[Dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# RNG snapshot
# ---------------------------------------------------------------------------

def snapshot_rng(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Capture every RNG stream the denoise loop can touch."""
    out: Dict[str, Any] = {
        "torch_cpu": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode(),
        "python": _py_random_state_to_json(random.getstate()),
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device)
        out["torch_cuda"] = base64.b64encode(cuda_state.numpy().tobytes()).decode()
        out["cuda_device"] = device.index if device.index is not None else torch.cuda.current_device()
    return out


def restore_rng(snap: Dict[str, Any], device: Optional[torch.device] = None) -> None:
    torch.set_rng_state(torch.frombuffer(
        bytearray(base64.b64decode(snap["torch_cpu"])), dtype=torch.uint8).clone())
    if "python" in snap:
        random.setstate(_py_random_state_from_json(snap["python"]))
    if "torch_cuda" in snap and torch.cuda.is_available():
        dev = device if device is not None else torch.device("cuda", snap.get("cuda_device", 0))
        torch.cuda.set_rng_state(torch.frombuffer(
            bytearray(base64.b64decode(snap["torch_cuda"])), dtype=torch.uint8).clone(), dev)


def _py_random_state_to_json(state: tuple) -> list:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _py_random_state_from_json(data: list) -> tuple:
    version, internal, gauss = data
    return (version, tuple(internal), gauss)


# ---------------------------------------------------------------------------
# Tensor codecs (M3's compression ladder starts here)
# ---------------------------------------------------------------------------

class _Codec:
    """Encode/decode one tensor as named parts. Part names are suffixes."""

    name = "none"

    def encode(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"": t.contiguous()}

    def decode(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        return parts[""]


class _Int8Codec(_Codec):
    """Symmetric per-tensor absmax int8. First rung of the M3 ladder."""

    name = "int8"

    def encode(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not t.is_floating_point():
            return {"": t.contiguous()}
        scale = t.abs().amax().clamp(min=1e-12).to(torch.float32) / 127.0
        q = (t.to(torch.float32) / scale).round().clamp(-127, 127).to(torch.int8)
        return {"": q, "._scale": scale.reshape(1), "._dtype": _dtype_tag(t.dtype)}

    def decode(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        q = parts[""]
        if "._scale" not in parts:
            return q
        scale = parts["._scale"].item()
        dtype = _dtype_from_tag(parts["._dtype"])
        return (q.to(torch.float32) * scale).to(dtype)


_CODECS: Dict[str, _Codec] = {c.name: c for c in (_Codec(), _Int8Codec())}

_DTYPE_TAGS = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2, torch.float64: 3}
_TAGS_DTYPE = {v: k for k, v in _DTYPE_TAGS.items()}


def _dtype_tag(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([_DTYPE_TAGS[dtype]], dtype=torch.int64)


def _dtype_from_tag(tag: torch.Tensor) -> torch.dtype:
    return _TAGS_DTYPE[int(tag.item())]


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """Everything needed to re-enter a neural world exactly where it was left.

    ``caches``  — named groups of per-block K/V caches (CPU tensors once captured),
                  e.g. {"kv": [...30 blocks...], "kv_mouse": [...], "cross": [...]}.
    ``tensors`` — named flat tensors: recent clean latents, streaming-VAE conv
                  cache entries, sliced conditioning, ... (host decides).
    ``meta``    — provenance: model id, checkpoint hash, resolution, timeline
                  DAG pointers (save_id / parent_id), so a save is self-describing.
    """

    meta: Dict[str, Any] = field(default_factory=dict)
    mode: str = "state"  # "state" | "reprime"
    caches: Dict[str, BlockCache] = field(default_factory=dict)
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    cursor: Dict[str, int] = field(default_factory=dict)
    actions: List[Any] = field(default_factory=list)
    rng: Dict[str, Any] = field(default_factory=dict)

    # -- capture / restore ---------------------------------------------------

    @classmethod
    def capture(
        cls,
        caches: Optional[Dict[str, BlockCache]] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        *,
        mode: str = "state",
        cursor: Optional[Dict[str, int]] = None,
        actions: Optional[List[Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> "WorldState":
        """Snapshot live cache groups / tensors (all copied to CPU)."""
        if mode not in ("state", "reprime"):
            raise ValueError(f"unknown mode {mode!r}")
        state = cls(mode=mode, meta=dict(meta or {}))
        if mode == "state":
            if not caches:
                raise ValueError("mode='state' requires at least one cache group")
            state.caches = {name: [_block_to_cpu(b) for b in blocks]
                            for name, blocks in caches.items()}
            if device is None:
                device = _blocks_device(next(iter(caches.values())))
        for name, t in (tensors or {}).items():
            state.tensors[name] = t.detach().to("cpu", copy=True)
            if device is None and t.is_cuda:
                device = t.device
        if mode == "reprime" and not state.tensors:
            raise ValueError("mode='reprime' requires the recent clean latents in `tensors`")
        state.cursor = dict(cursor or {})
        state.actions = list(actions or [])
        state.rng = snapshot_rng(device)
        state.meta.setdefault("wsave_version", WSAVE_VERSION)
        return state

    def restore_caches(
        self,
        targets: Dict[str, BlockCache],
        *,
        device: Optional[torch.device] = None,
        restore_rng_state: bool = True,
    ) -> None:
        """Write cache groups back into live pipeline structures (mode='state').

        Reprime saves have no caches to write; the host adapter replays the
        model's own context-caching pass instead and then only restores RNG.
        """
        if self.mode != "state":
            raise ValueError("restore_caches() only applies to mode='state'; "
                             "use the host adapter's reprime path for reprime saves")
        missing = set(self.caches) - set(targets)
        if missing:
            raise KeyError(f"pipeline is missing cache groups: {sorted(missing)}")
        if device is None:
            device = _blocks_device(next(iter(targets.values())))
        for name, blocks in self.caches.items():
            _write_blocks(blocks, targets[name], device)
        if restore_rng_state and self.rng:
            restore_rng(self.rng, device)

    # -- introspection ---------------------------------------------------------

    def nbytes(self) -> Dict[str, int]:
        """Tensor payload bytes per component (inspector / M3 accounting)."""
        out: Dict[str, int] = {}
        for name, blocks in self.caches.items():
            out[f"cache.{name}"] = sum(
                v.numel() * v.element_size()
                for b in blocks for v in b.values() if isinstance(v, torch.Tensor))
        for name, t in self.tensors.items():
            out[f"tensor.{name}"] = t.numel() * t.element_size()
        out["__total__"] = sum(out.values())
        return out


# ---------------------------------------------------------------------------
# capture/restore helpers
# ---------------------------------------------------------------------------

def _blocks_device(blocks: BlockCache) -> torch.device:
    for b in blocks:
        for v in b.values():
            if isinstance(v, torch.Tensor):
                return v.device
    return torch.device("cpu")


def _block_to_cpu(block: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in block.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to("cpu", copy=True)
        elif isinstance(v, bool):
            out[k] = torch.tensor([v])
        else:
            raise TypeError(f"unexpected cache entry {k}: {type(v)}")
    return out


def _write_blocks(src_blocks: BlockCache, dst_blocks: BlockCache, device) -> None:
    if len(src_blocks) != len(dst_blocks):
        raise ValueError(f"block count mismatch: save has {len(src_blocks)}, "
                         f"pipeline has {len(dst_blocks)}")
    for src, dst in zip(src_blocks, dst_blocks):
        for k, v in src.items():
            if k not in dst:
                raise KeyError(f"pipeline cache block missing key {k!r}")
            if isinstance(dst[k], torch.Tensor):
                if dst[k].shape != v.shape:
                    raise ValueError(f"shape mismatch for {k}: save {tuple(v.shape)} "
                                     f"vs pipeline {tuple(dst[k].shape)}")
                dst[k].copy_(v.to(device))
            elif isinstance(dst[k], bool):
                dst[k] = bool(v.item())


# ---------------------------------------------------------------------------
# .wsave container I/O
# ---------------------------------------------------------------------------

def save_wsave(state: WorldState, path: str, *, quant: str = "none") -> Dict[str, Any]:
    """Write a WorldState to a ``.wsave`` file. Returns the header written.

    Only cache K/V payloads are quantized; cursors, latents and VAE cache
    entries are stored exactly (they are small and fidelity-critical).
    """
    codec = _CODECS[quant]
    tensors: Dict[str, torch.Tensor] = {}
    layout: Dict[str, Any] = {
        "cache_groups": {name: len(blocks) for name, blocks in state.caches.items()},
        "tensor_names": sorted(state.tensors),
    }

    def put(name: str, t: torch.Tensor, quantize: bool) -> None:
        parts = (codec if quantize else _CODECS["none"]).encode(t)
        for suffix, part in parts.items():
            tensors[name + suffix] = part

    for group, blocks in state.caches.items():
        for i, b in enumerate(blocks):
            for k, v in b.items():
                put(f"cache.{group}.{i:03d}.{k}", v, quantize=k in _PAYLOAD_KEYS)
    for name, t in state.tensors.items():
        put(f"tensor.{name}", t, quantize=False)

    header = {
        "wsave_version": WSAVE_VERSION,
        "mode": state.mode,
        "quant": codec.name,
        "cursor": state.cursor,
        "layout": layout,
        "meta": state.meta,
    }
    blob = st_save(tensors)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(_HEADER_NAME, json.dumps(header, indent=2))
        zf.writestr(_TENSORS_NAME, blob)
        zf.writestr(_ACTIONS_NAME, json.dumps(state.actions))
        zf.writestr(_RNG_NAME, json.dumps(state.rng))
    return header


def load_wsave(path: str) -> WorldState:
    with zipfile.ZipFile(path, "r") as zf:
        header = json.loads(zf.read(_HEADER_NAME))
        if header["wsave_version"] > WSAVE_VERSION:
            raise ValueError(f"save file version {header['wsave_version']} is newer "
                             f"than this library ({WSAVE_VERSION})")
        tensors = st_load(zf.read(_TENSORS_NAME))
        actions = json.loads(zf.read(_ACTIONS_NAME))
        rng = json.loads(zf.read(_RNG_NAME))

    codec = _CODECS[header["quant"]]

    def take(prefix: str, quantized: bool) -> torch.Tensor:
        parts = {name[len(prefix):]: t for name, t in tensors.items()
                 if name == prefix or name.startswith(prefix + ".")}
        return (codec if quantized else _CODECS["none"]).decode(parts)

    state = WorldState(meta=header["meta"], mode=header["mode"],
                       cursor=header["cursor"], actions=actions, rng=rng)
    for group, nblocks in header["layout"]["cache_groups"].items():
        state.caches[group] = [
            _read_block(tensors, take, group, i) for i in range(nblocks)]
    for name in header["layout"]["tensor_names"]:
        state.tensors[name] = take(f"tensor.{name}", quantized=False)
    return state


def _read_block(tensors, take, group: str, i: int) -> Dict[str, torch.Tensor]:
    prefix = f"cache.{group}.{i:03d}."
    keys = {name[len(prefix):].split(".")[0] for name in tensors if name.startswith(prefix)}
    return {k: take(prefix + k, quantized=k in _PAYLOAD_KEYS) for k in sorted(keys)}
