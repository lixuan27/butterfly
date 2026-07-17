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


# Mission「守梦远征」— the default objective. Explore N blocks (they become the
# original timeline), then fate re-rolls (THE TEAR) and you must hold the world:
# a block whose similarity-vs-ghost falls to 0 (worse than doing nothing against
# chaos) is a strike; MISSION_STRIKES strikes tear the dream. Survive the full
# hold and the dream is held — graded by the usual duel score.
MISSION_EXPLORE_BLOCKS = 10
MISSION_STRIKES = 3
# Tear line, calibrated on the 184931 e2e measurement: a player who merely
# mimics their ghost sits at similarity ~0 (score 0.043) and must NOT strike
# out; a strike means actively pushing the world further than chaos itself
# (dist > 1.25x the measured chaos baseline).
MISSION_TEAR_LINE = -0.25


@dataclass
class MissionState:
    phase: str = "explore"            # explore | hold
    # None until the first step: capturing at t=0 is impossible (the streaming
    # VAE's conv cache only exists after the first decode — 184937 root cause),
    # so the home anchor drops lazily after the player's first move.
    anchor_id: Optional[str] = None
    strikes: int = 0


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
        self.mission: Optional[MissionState] = None
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
        self.mission = None            # a fresh dream owes nothing to an old mission
        self._current_anchor = None
        return {"type": "started", **self._status()}

    def step(self, action: Dict[str, Any]):
        """Advance one block. Returns (frames_live, ghost_frame_or_None, msg)."""
        if self.mode == "idle":
            raise RuntimeError("start a game first")
        if self.blocks_left <= 0:
            self.mode = "idle"
            msg = {"type": "collapsed", **self._status()}
            if self.mission is not None:
                msg = self._mission_msg(msg)
            return None, None, msg
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
        if self.mission is not None and self.mission.anchor_id is None \
                and self.mode == "free":
            # first move done -> the streaming caches exist -> drop home
            m, self.mission = self.mission, None    # bypass the mission guard
            try:
                a = self.drop_anchor("home")
            finally:
                self.mission = m
            m.anchor_id = a["save_id"]
            msg["home_anchor"] = a["save_id"]
        if self.mission is not None:
            msg = self._mission_msg(msg)
        return live_u8, ghost_u8, msg

    def _similarity(self, dist: float) -> float:
        base = self._chaos_mean(max(1, self.duel.block + 1))
        return max(0.0, 1.0 - dist / max(base, 1e-9))

    def drop_anchor(self, label: str) -> Dict[str, Any]:
        from savepoint.state import save_wsave
        self._mission_guard()
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
        self._mission_guard()
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
        self._mission_guard()
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
        self._mission_guard()
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

    # -- mission「守梦远征」— the default objective ---------------------------

    def start_mission(self, image, seed: int) -> Dict[str, Any]:
        """New dream with an explicit goal: explore, survive the tear, hold."""
        self.mission = None            # clear any stale mission before reset
        self.new_game(image, seed)
        self.mission = MissionState()  # home anchor drops after the first move
        return {"type": "mission_started", "anchor": None,
                "explore_blocks": MISSION_EXPLORE_BLOCKS,
                "strikes_allowed": MISSION_STRIKES, **self._status()}

    def mission_tear(self, seed: int) -> Dict[str, Any]:
        """Fate re-rolls; the hold phase begins."""
        m = self.mission
        if not m or m.phase != "explore":
            raise RuntimeError("no mission is exploring")
        ghost = self.ghosts.get(m.anchor_id)
        if not ghost or not ghost.latents:
            raise RuntimeError("walk a little first — the tear needs a timeline to threaten")
        self.mission = None            # internal duel start bypasses the guard
        try:
            msg = self.start_duel(m.anchor_id, seed)
        finally:
            self.mission = m
        m.phase, m.strikes = "hold", 0
        return {**msg, "type": "mission_tear",
                "strikes_allowed": MISSION_STRIKES}

    def abandon_mission(self) -> Dict[str, Any]:
        m = self.mission
        if not m:
            raise RuntimeError("no mission to abandon")
        self.mission = None
        if self.mode == "duel" and self.duel and self.duel.anchor_id == m.anchor_id:
            self.mode, self.duel = "free", None
            self.ghosts[m.anchor_id] = Ghost()
        return {"type": "mission_abandoned", **self._status()}

    def _mission_guard(self) -> None:
        if self.mission is not None:
            raise RuntimeError("the mission holds this dream — "
                               "hold it to the end or abandon it first")

    def _mission_msg(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay mission progress / verdicts onto the base message stream."""
        m = self.mission
        if m.anchor_id is None:        # pre-anchor step (shouldn't happen)
            return msg
        if msg["type"] == "stepped" and m.phase == "explore":
            played = len(self.ghosts[m.anchor_id].actions)
            tear_in = max(0, MISSION_EXPLORE_BLOCKS - played)
            msg["mission"] = {"phase": "explore", "tear_in": tear_in,
                              "tear_now": tear_in == 0}
        elif msg["type"] == "duel_tick" and m.phase == "hold":
            base = self._chaos_mean(max(1, msg["block"]))
            sim_raw = 1.0 - msg["dist"] / max(base, 1e-9)
            if sim_raw < MISSION_TEAR_LINE:
                m.strikes += 1
            if m.strikes >= MISSION_STRIKES:
                d = self.duel
                self.mode, self.duel, self.mission = "free", None, None
                self.ghosts[d.anchor_id] = Ghost()
                mean_d = sum(d.dists) / len(d.dists)
                return {"type": "mission_end", "won": False, "reason": "torn",
                        "strikes": MISSION_STRIKES, "held": d.block,
                        "total": msg["total"], "mean_dist": mean_d,
                        "chaos_baseline": self._chaos_mean(len(d.dists)),
                        **self._status()}
            msg["mission"] = {"phase": "hold", "strikes": m.strikes,
                              "strikes_allowed": MISSION_STRIKES,
                              "sim_raw": sim_raw,
                              "held": msg["block"], "total": msg["total"]}
        elif msg["type"] == "duel_end" and m.phase == "hold":
            # the final block ends the duel before its tick — judge it here too
            dists = msg.get("dists") or []
            if dists:
                base = self._chaos_mean(len(dists))
                if 1.0 - dists[-1] / max(base, 1e-9) < MISSION_TEAR_LINE:
                    m.strikes += 1
            self.mission = None
            if m.strikes >= MISSION_STRIKES:
                msg = {**msg, "type": "mission_end", "won": False,
                       "reason": "torn", "strikes": m.strikes,
                       "held": len(dists), "total": len(dists)}
            else:
                msg = {**msg, "type": "mission_end", "won": True,
                       "reason": "held", "strikes": m.strikes}
        elif msg["type"] == "collapsed":
            self.mission = None
            msg = {**msg, "type": "mission_end", "won": False,
                   "reason": "collapsed", "strikes": m.strikes}
        return msg

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
        self._mission_guard()
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
