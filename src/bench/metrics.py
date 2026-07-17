"""Host-agnostic measurement primitives for SaveBench.

Conventions
-----------
* A *video* is a float tensor ``[T, C, H, W]`` in [0, 1].
* A *feature trajectory* is ``[T, D]`` (e.g. per-frame DINO embeddings).
* A *divergence curve* is ``[T]``: distance between two rollouts at each step
  after a common branch point. Curves start at the first generated frame
  post-branch (t=0 is the first frame where the two runs could differ).
"""

from __future__ import annotations

import math
import zipfile
from typing import Callable, Dict, List, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# frame-level distances (M1: resume vs uninterrupted; M2: branch pairs)
# ---------------------------------------------------------------------------


def per_frame_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[T,...] x [T,...] -> [T] mean squared error per frame."""
    _check_same_shape(a, b)
    return (a.float() - b.float()).pow(2).flatten(1).mean(dim=1)


def per_frame_psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """[T] PSNR per frame; +inf where frames are identical."""
    mse = per_frame_mse(a, b)
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=0))  # inf on mse=0 is intended


def feature_distance(fa: torch.Tensor, fb: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    """[T,D] x [T,D] -> [T]. Cosine distance (1-cos) or L2."""
    _check_same_shape(fa, fb)
    if kind == "cosine":
        return 1.0 - torch.nn.functional.cosine_similarity(fa.float(), fb.float(), dim=-1)
    if kind == "l2":
        return (fa.float() - fb.float()).norm(dim=-1)
    raise ValueError(f"unknown kind {kind!r}")


def _check_same_shape(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


# ---------------------------------------------------------------------------
# M2: fork divergence — half-life and controllability SNR
# ---------------------------------------------------------------------------


def pairwise_divergence(
    rollouts: Sequence[torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = per_frame_mse,
) -> torch.Tensor:
    """N same-length rollouts from one branch point -> mean divergence curve [T]
    over all N*(N-1)/2 pairs."""
    n = len(rollouts)
    if n < 2:
        raise ValueError("need >= 2 rollouts to measure divergence")
    curves = [distance(rollouts[i], rollouts[j]) for i in range(n) for j in range(i + 1, n)]
    return torch.stack(curves).mean(dim=0)


def half_life(
    curve: torch.Tensor,
    threshold: float,
    dt: float = 1.0,
) -> Optional[float]:
    """First time the divergence curve crosses `threshold`, linearly
    interpolated between steps; None if it never crosses. This is T½ when
    threshold is set to half the saturation plateau (see `saturation_level`)."""
    above = (curve >= threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return None
    i = int(above[0].item())
    if i == 0:
        return 0.0
    c0, c1 = float(curve[i - 1]), float(curve[i])
    frac = 0.0 if c1 == c0 else (threshold - c0) / (c1 - c0)
    return (i - 1 + frac) * dt

def saturation_level(curve: torch.Tensor, tail_frac: float = 0.25) -> float:
    """Plateau estimate: mean of the trailing `tail_frac` of the curve."""
    t = max(1, int(len(curve) * tail_frac))
    return float(curve[-t:].mean())


def controllability_snr(
    action_curve: torch.Tensor,
    noise_curve: torch.Tensor,
    horizon: Optional[int] = None,
) -> Dict[str, float]:
    """Behavioral controllability: how much faster do *actions* separate two
    worlds than *noise* does?

    Both curves share the same branch point and time base. Returns the areas
    under each curve up to `horizon` and their ratio (>1 means actions steer
    the world above the chaos floor)."""
    T = min(len(action_curve), len(noise_curve))
    if horizon is not None:
        T = min(T, horizon)
    a = float(action_curve[:T].sum())
    n = float(noise_curve[:T].sum())
    return {
        "action_auc": a,
        "noise_auc": n,
        "snr": a / n if n > 0 else math.inf,
        "horizon": T,
    }


# ---------------------------------------------------------------------------
# M1/M3: save-file forensics — size vs fidelity
# ---------------------------------------------------------------------------


def wsave_size_report(path: str) -> Dict[str, int]:
    """Bytes per member of a .wsave container (uncompressed sizes)."""
    with zipfile.ZipFile(path) as zf:
        sizes = {i.filename: i.file_size for i in zf.infolist()}
    sizes["__total__"] = sum(sizes.values())
    return sizes


def fidelity_size_curve(
    points: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Sort {size_bytes, fidelity_*}-dicts by size for the M3 rate-distortion
    style plot; purely a convenience shaper for logging."""
    return sorted(points, key=lambda p: p["size_bytes"])
