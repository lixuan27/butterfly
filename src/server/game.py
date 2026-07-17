"""Butterfly game logic: the state machine between the browser and the world model.

One live world per GPU. The game wraps the StreamingHost with:
  * anchors    — up to 3 named .wsave savepoints (the timeline tree)
  * recording  — after an anchor, the played segment becomes that anchor's
                 "original timeline" (ghost) : frames + latents + actions
  * duel       — restore an anchor with a re-rolled noise stream (chaos!) and
                 replay the same duration; per-block similarity vs the ghost
                 is the score. Beating the do-nothing chaos baseline = skill.
  * dream life — the RoPE table budget (360 latent frames) is the session clock.

Pure-python; transport-agnostic (the FastAPI layer just forwards messages).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

MAX_ANCHORS = 3
GRADES = ((0.35, "S"), (0.20, "A"), (0.05, "B"), (-1e9, "C"))
# Butterfly grading: amplification of your ONE flap vs a full re-roll of fate.
BFLY_GRADES = ((1.25, "S"), (0.8, "A"), (0.35, "B"), (-1e9, "C"))


@dataclass
class Ghost:
    """The original timeline recorded after an anchor."""

    frames: List[torch.Tensor] = field(default_factory=list)   # uint8 [12,H,W,3] per block
    latents: List[torch.Tensor] = field(default_factory=list)  # [16,3,44,80] per block
    actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DuelState:
    anchor_id: str
    block: int = 0
    dists: List[float] = field(default_factory=list)


@dataclass
class ButterflyState:
    """One-flap divergence race: same fate (saved RNG), one changed action."""

    anchor_id: str
    block: int = 0
    flap_block: Optional[int] = None
    dists: List[float] = field(default_factory=list)  # from the flap onward


class ButterflyGame:
    def __init__(self, host, timeline, save_dir: str,
                 chaos_baseline_path: Optional[str] = None):
        self.host = host
        self.timeline = timeline
        self.save_dir = save_dir
        self.mode = "idle"            # idle | free | duel | butterfly
        self.anchors: List[str] = []  # save_ids, oldest first
        self.ghosts: Dict[str, Ghost] = {}
        self.duel: Optional[DuelState] = None
        self.bfly: Optional[ButterflyState] = None
        self.chaos_curve = self._load_chaos_baseline(chaos_baseline_path)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _load_chaos_baseline(path: Optional[str]) -> List[float]:
        """Mean noise-fork latent-MSE curve from the M2 run (per pixel frame)."""
        if path and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)["noise"]["curve"]
        return []

    def _chaos_mean(self, n_blocks: int) -> float:
        """Baseline mean latent distance over a duel horizon of n blocks."""
        if not self.chaos_curve:
            return 0.25  # fallback: retro-fit from M2 plateau
        horizon = min(len(self.chaos_curve), n_blocks * 12)
        return sum(self.chaos_curve[:horizon]) / max(1, horizon)

    @property
    def blocks_left(self) -> int:
        return (self.host.max_latent_frames - self.host.cursor["latent_frame"]) \
            // self.host.num_frame_per_block

    def _status(self) -> Dict[str, Any]:
        return {"mode": self.mode, "blocks_left": self.blocks_left,
                "cursor": self.host.cursor, "anchors": self.anchors,
                "current": getattr(self, "_current_anchor", None)}

    # -- commands ------------------------------------------------------------

    def new_game(self, image, seed: int) -> Dict[str, Any]:
        self.host.prime(image, seed=seed)
        self.mode = "free"
        self.anchors, self.ghosts, self.duel, self.bfly = [], {}, None, None
        self._current_anchor = None
        return {"type": "started", **self._status()}

    def step(self, action: Dict[str, Any]):
        """Advance one block. Returns (frames_live, ghost_frame_or_None, msg)."""
        if self.mode == "idle":
            raise RuntimeError("start a game first")
        if self.blocks_left <= 0:
            self.mode = "idle"
            return None, None, {"type": "collapsed", **self._status()}
        flap = bool(action.get("flap"))
        action = {"keys": action.get("keys", []), "mouse": action.get("mouse", [0.0, 0.0])}
        if self.mode == "butterfly":
            b = self.bfly
            if flap and b.flap_block is None:
                b.flap_block = b.block  # the one wing-beat: the player's action goes in
            else:
                action = self.ghosts[b.anchor_id].actions[b.block]  # faithful replay
        result = self.host.step(action)
        live_u8 = (result.frames.permute(0, 2, 3, 1) * 255).clamp(0, 255) \
            .to(torch.uint8).cpu()  # JPEG encoding is host-side

        ghost_u8, msg = None, None
        if self.mode == "duel":
            d = self.duel
            ghost = self.ghosts[d.anchor_id]
            # pixel-space MSE in [0,1] — same space & scale as the M2 chaos baseline
            dist = float(((live_u8.float() - ghost.frames[d.block].float()) / 255.0)
                         .pow(2).mean())
            d.dists.append(dist)
            ghost_u8 = ghost.frames[d.block]
            d.block += 1
            similarity = self._similarity(dist)
            if d.block >= len(ghost.latents):
                msg = self._end_duel()
            else:
                msg = {"type": "duel_tick", "block": d.block,
                       "total": len(ghost.latents), "dist": dist,
                       "similarity": similarity, **self._status()}
        elif self.mode == "butterfly":
            b = self.bfly
            ghost = self.ghosts[b.anchor_id]
            dist = float(((live_u8.float() - ghost.frames[b.block].float()) / 255.0)
                         .pow(2).mean())
            if b.flap_block is not None:
                b.dists.append(dist)
            ghost_u8 = ghost.frames[b.block]
            b.block += 1
            base = self._chaos_mean(max(1, b.block))
            similarity = max(0.0, 1.0 - dist / max(base, 1e-9))
            if b.block >= len(ghost.latents):
                msg = self._end_butterfly()
            else:
                msg = {"type": "butterfly_tick", "block": b.block,
                       "total": len(ghost.latents), "dist": dist,
                       "similarity": similarity,
                       "flapped": b.flap_block is not None,
                       "flap_block": b.flap_block, **self._status()}
        else:
            # free mode: record into the current anchor's ghost
            if self._current_anchor is not None:
                g = self.ghosts[self._current_anchor]
                g.frames.append(live_u8)
                g.latents.append(result.latents.float().cpu())
                g.actions.append(action)
            msg = {"type": "stepped", "stats": result.stats, **self._status()}
        return live_u8, ghost_u8, msg

    def _similarity(self, dist: float) -> float:
        base = self._chaos_mean(max(1, self.duel.block + 1))
        return max(0.0, 1.0 - dist / max(base, 1e-9))

    def drop_anchor(self, label: str) -> Dict[str, Any]:
        from savepoint.state import save_wsave
        if self.mode != "free":
            raise RuntimeError("anchors only in free play")
        if len(self.anchors) >= MAX_ANCHORS:
            raise RuntimeError(f"all {MAX_ANCHORS} anchor slots used")
        state = self.host.capture(mode="state")
        node = self.timeline.add(
            path="pending", latent_frame=self.host.cursor["latent_frame"],
            mode="state", parent_id=self._current_anchor, label=label)
        fname = f"{node.save_id}.wsave"
        save_wsave(state, os.path.join(self.save_dir, fname))
        self.timeline.nodes[node.save_id].path = fname
        self.timeline._flush()
        self.anchors.append(node.save_id)
        self.ghosts[node.save_id] = Ghost()
        self._current_anchor = node.save_id
        return {"type": "anchored", "save_id": node.save_id,
                "bytes": os.path.getsize(os.path.join(self.save_dir, fname)),
                **self._status()}

    def rewind(self, anchor_id: str) -> Dict[str, Any]:
        from savepoint.state import load_wsave
        if anchor_id not in self.ghosts:
            raise KeyError(f"unknown anchor {anchor_id}")
        self.host.restore(load_wsave(os.path.join(
            self.save_dir, self.timeline.nodes[anchor_id].path)))
        # rewinding truncates that anchor's ghost: a new original timeline begins
        self.ghosts[anchor_id] = Ghost()
        self._current_anchor = anchor_id
        self.mode = "free"
        self.duel = None
        self.bfly = None
        return {"type": "rewound", "anchor": anchor_id, **self._status()}

    def start_duel(self, anchor_id: str, seed: int) -> Dict[str, Any]:
        from savepoint.state import load_wsave
        ghost = self.ghosts.get(anchor_id)
        if not ghost or not ghost.latents:
            raise RuntimeError("play some blocks after this anchor first — "
                               "the duel needs an original timeline to defend")
        self.host.restore(load_wsave(os.path.join(
            self.save_dir, self.timeline.nodes[anchor_id].path)))
        self.host._rng.manual_seed(seed)  # re-roll chance: chaos begins
        self.mode = "duel"
        self.duel = DuelState(anchor_id=anchor_id)
        self.bfly = None
        self._current_anchor = anchor_id
        return {"type": "duel_started", "anchor": anchor_id,
                "total": len(ghost.latents),
                "ghost_actions": ghost.actions, **self._status()}

    def _end_duel(self) -> Dict[str, Any]:
        d = self.duel
        mean_d = sum(d.dists) / len(d.dists)
        base = self._chaos_mean(len(d.dists))
        score = max(-1.0, min(1.0, 1.0 - mean_d / max(base, 1e-9)))
        grade = next(g for thr, g in GRADES if score >= thr)
        self.mode = "free"
        self.duel = None
        # the duel branch is now the live timeline; its anchor ghost resets
        self.ghosts[d.anchor_id] = Ghost()
        return {"type": "duel_end", "score": score, "grade": grade,
                "mean_dist": mean_d, "chaos_baseline": base,
                "dists": d.dists, **self._status()}

    # -- butterfly「扇翅」 ----------------------------------------------------

    def start_butterfly(self, anchor_id: str) -> Dict[str, Any]:
        from savepoint.state import load_wsave
        ghost = self.ghosts.get(anchor_id)
        if not ghost or not ghost.latents:
            raise RuntimeError("play some blocks after this anchor first — "
                               "the flap needs an original timeline to disturb")
        # Same fate: the .wsave brings the RNG back, so a faithful replay is
        # bit-exact and every ripple on screen is caused by the one flap alone.
        self.host.restore(load_wsave(os.path.join(
            self.save_dir, self.timeline.nodes[anchor_id].path)))
        self.mode = "butterfly"
        self.duel = None
        self.bfly = ButterflyState(anchor_id=anchor_id)
        self._current_anchor = anchor_id
        return {"type": "butterfly_started", "anchor": anchor_id,
                "total": len(ghost.latents),
                "ghost_actions": ghost.actions, **self._status()}

    def _end_butterfly(self) -> Dict[str, Any]:
        b = self.bfly
        self.mode = "free"
        self.bfly = None
        # the flapped branch is now the live timeline; the anchor's ghost resets
        self.ghosts[b.anchor_id] = Ghost()
        if b.flap_block is None or not b.dists:
            return {"type": "butterfly_end", "score": 0.0, "grade": "C",
                    "amp": 0.0, "flap_block": None, "mean_dist": 0.0,
                    "chaos_baseline": self._chaos_mean(1), "dists": [],
                    **self._status()}
        mean_d = sum(b.dists) / len(b.dists)
        base = self._chaos_mean(len(b.dists))
        amp = mean_d / max(base, 1e-9)
        grade = next(g for thr, g in BFLY_GRADES if amp >= thr)
        return {"type": "butterfly_end", "score": amp, "grade": grade,
                "amp": amp, "flap_block": b.flap_block, "mean_dist": mean_d,
                "chaos_baseline": base, "dists": b.dists, **self._status()}

    # -- relay「传梦」 --------------------------------------------------------

    def export_info(self, anchor_id: str) -> Dict[str, Any]:
        node = self.timeline.nodes.get(anchor_id)
        if node is None:
            raise KeyError(f"unknown anchor {anchor_id}")
        path = os.path.join(self.save_dir, node.path)
        return {"type": "export", "save_id": anchor_id, "path": path,
                "bytes": os.path.getsize(path), "label": node.label,
                "latent_frame": node.latent_frame}

    def import_save(self, path: str, label: str = "") -> Dict[str, Any]:
        import shutil
        from savepoint.state import load_wsave
        if self.mode in ("duel", "butterfly"):
            raise RuntimeError("finish the current round first")
        if len(self.anchors) >= MAX_ANCHORS:
            raise RuntimeError(f"all {MAX_ANCHORS} anchor slots used")
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        state = load_wsave(path)  # validate before adopting someone's dream
        node = self.timeline.add(
            path="pending", latent_frame=int(state.cursor.get("latent_frame", 0)),
            mode=state.mode, parent_id=None, label=label or "relayed dream")
        fname = f"{node.save_id}.wsave"
        dest = os.path.join(self.save_dir, fname)
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copy2(path, dest)
        self.timeline.nodes[node.save_id].path = fname
        self.timeline._flush()
        self.anchors.append(node.save_id)
        self.ghosts[node.save_id] = Ghost()
        return {"type": "imported", "save_id": node.save_id,
                "bytes": os.path.getsize(dest), "label": node.label,
                **self._status()}
