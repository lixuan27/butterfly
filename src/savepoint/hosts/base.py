"""The StreamingHost protocol: what a world model must expose to be saveable.

Design notes
------------
The host — not the model — owns everything the model's own inference script
keeps in Python local variables: the frame cursor, the action history, the
per-segment conditioning, the noise stream. That is exactly the state that
evaporates when an inference process dies, and exactly what `.wsave` must
capture alongside the model's KV/VAE caches.

Every method is synchronous and single-world; the demo server multiplexes
worlds by holding several hosts (or by save/load swapping on one GPU).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..state import WorldState


@dataclass
class StepResult:
    """One streaming step's output."""

    frames: torch.Tensor           # [n, C, H, W] decoded pixels in [0,1]
    latents: torch.Tensor          # [n_lat, C_lat, h, w] clean latents for this step
    frame_index: int               # index of the first returned frame in the episode
    stats: Dict[str, float]        # timing / debug info (step_ms, fps, ...)


class StreamingHost(abc.ABC):
    """Uniform facade over one streaming interactive world model."""

    #: human-readable id, e.g. "matrix-game-2/base_distill"
    model_id: str = "unknown"

    # -- lifecycle -----------------------------------------------------------

    @abc.abstractmethod
    def prime(self, image: Any, *, seed: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """Start a fresh world from a conditioning image (and internal prompt),
        fully deterministic under `seed`."""

    @abc.abstractmethod
    def step(self, action: Dict[str, Any], *, n_latent_frames: int = 1) -> StepResult:
        """Advance the world by one block under `action` (host-specific schema,
        e.g. {"keys": [...], "mouse": [dx, dy]})."""

    # -- savepoint ------------------------------------------------------------

    @abc.abstractmethod
    def capture(self, *, mode: str = "state") -> WorldState:
        """Snapshot the complete runtime state (mode='state') or the minimal
        re-priming payload (mode='reprime')."""

    @abc.abstractmethod
    def restore(self, state: WorldState) -> None:
        """Re-enter a captured world. mode='state' writes caches back;
        mode='reprime' replays the model's own context-caching pass on the
        saved latents. Both must leave the host ready for `step()`."""

    # -- introspection ---------------------------------------------------------

    @property
    @abc.abstractmethod
    def cursor(self) -> Dict[str, int]:
        """Current position, e.g. {"latent_frame": 12, "pixel_frame": 45}."""

    @property
    @abc.abstractmethod
    def action_history(self) -> List[Dict[str, Any]]:
        """Actions applied since prime(), JSON-serializable."""

    # -- optional capabilities -------------------------------------------------

    def state_components(self) -> Dict[str, int]:
        """Byte sizes of each runtime-state component (for the inspector UI &
        M3 accounting). Override in concrete hosts."""
        return {}
