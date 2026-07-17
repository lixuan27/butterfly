"""CPU unit tests for ButterflyGame: anchor/duel/butterfly/relay state machine.

Uses a deterministic FakeHost (no noise, no GPU): the world is a single float
that evolves as a function of the action, frames encode it as a constant image.
Faithful replay is therefore bit-exact — exactly the property the real host
guarantees via saved RNG — so butterfly pre-flap distances must be 0.
"""

import os
import sys
import shutil
import tempfile
import unittest

import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJ, "src"))

from savepoint.hosts.base import StepResult          # noqa: E402
from savepoint.state import WorldState               # noqa: E402
from savepoint.timeline import Timeline              # noqa: E402
from server.game import ButterflyGame                # noqa: E402

STEP = {"w": 1.0, "a": 2.0, "d": 0.25}


class FakeHost:
    model_id = "fake"
    num_frame_per_block = 3
    max_latent_frames = 60

    def __init__(self):
        self._rng = torch.Generator().manual_seed(0)
        self._lat = 0
        self._world = 0.0
        self._history = []

    def prime(self, image, *, seed, meta=None):
        self._lat, self._world, self._history = 1, float(seed % 7), []

    def step(self, action, *, n_latent_frames=1):
        self._world += STEP.get((action.get("keys") or ["_"])[0], 0.5)
        self._lat += self.num_frame_per_block
        self._history.append(action)
        frames = torch.full((3, 3, 8, 8), (self._world % 10) / 10.0)
        latents = torch.full((1, 4, 2, 2), self._world)
        return StepResult(frames=frames, latents=latents,
                          frame_index=self._lat, stats={"fps": 999.0})

    def capture(self, *, mode="state"):
        return WorldState(meta={"model": self.model_id}, mode=mode,
                          cursor={"latent_frame": self._lat},
                          tensors={"world": torch.tensor([self._world])},
                          actions=list(self._history), rng={})

    def restore(self, state):
        self._lat = int(state.cursor["latent_frame"])
        self._world = float(state.tensors["world"][0])
        self._history = list(state.actions)

    @property
    def cursor(self):
        return {"latent_frame": self._lat}

    @property
    def action_history(self):
        return self._history

    def state_components(self):
        return {"__total__": 8}


A_W = {"keys": ["w"], "mouse": [0.0, 0.0]}
A_A = {"keys": ["a"], "mouse": [0.0, 0.0]}


class GameTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="savepoint_test_")
        self.game = ButterflyGame(FakeHost(), Timeline(self.dir), self.dir)
        self.game.new_game(None, seed=3)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _anchor_and_ghost(self, n=3):
        aid = self.game.drop_anchor("t")["save_id"]
        for _ in range(n):
            live, ghost, msg = self.game.step(dict(A_W))
            self.assertEqual(msg["type"], "stepped")
            self.assertIsNone(ghost)
        return aid

    def test_duel_faithful_replay_is_perfect(self):
        aid = self._anchor_and_ghost(3)
        ds = self.game.start_duel(aid, seed=99)
        self.assertEqual(ds["type"], "duel_started")
        self.assertEqual(ds["total"], 3)
        end = None
        for _ in range(3):
            live, ghost, msg = self.game.step(dict(A_W))
            self.assertIsNotNone(ghost)
            if msg["type"] == "duel_end":
                end = msg
        self.assertIsNotNone(end)
        # FakeHost has no noise: mirroring the ghost's action is a perfect hold
        self.assertEqual(end["mean_dist"], 0.0)
        self.assertEqual(end["grade"], "S")
        self.assertEqual(self.game.mode, "free")

    def test_butterfly_one_flap_diverges(self):
        aid = self._anchor_and_ghost(4)
        bs = self.game.start_butterfly(aid)
        self.assertEqual(bs["type"], "butterfly_started")
        # block 0: no flap -> faithful replay, dist must be exactly 0
        _, ghost, t0 = self.game.step({"keys": ["a"], "mouse": [0, 0]})  # keys ignored
        self.assertEqual(t0["type"], "butterfly_tick")
        self.assertEqual(t0["dist"], 0.0)
        self.assertFalse(t0["flapped"])
        # block 1: the one flap, with a different action
        _, _, t1 = self.game.step({**A_A, "flap": True})
        self.assertTrue(t1["flapped"])
        self.assertEqual(t1["flap_block"], 1)
        self.assertGreater(t1["dist"], 0.0)
        # block 2: a second flap attempt is ignored (replay resumes)
        _, _, t2 = self.game.step({**A_A, "flap": True})
        self.assertEqual(t2["flap_block"], 1)
        # block 3: end
        _, _, end = self.game.step(dict(A_W))
        self.assertEqual(end["type"], "butterfly_end")
        self.assertEqual(end["flap_block"], 1)
        self.assertEqual(len(end["dists"]), 3)      # flap block onward
        self.assertGreater(end["amp"], 0.0)
        self.assertIn(end["grade"], "SABC")
        self.assertEqual(self.game.mode, "free")

    def test_butterfly_never_flapped_scores_zero(self):
        aid = self._anchor_and_ghost(2)
        self.game.start_butterfly(aid)
        for _ in range(1):
            self.game.step(dict(A_W))
        _, _, end = self.game.step(dict(A_W))
        self.assertEqual(end["type"], "butterfly_end")
        self.assertIsNone(end["flap_block"])
        self.assertEqual(end["score"], 0.0)
        self.assertEqual(end["grade"], "C")

    def test_butterfly_requires_ghost(self):
        aid = self.game.drop_anchor("empty")["save_id"]
        with self.assertRaises(RuntimeError):
            self.game.start_butterfly(aid)

    def test_relay_export_import_roundtrip(self):
        aid = self._anchor_and_ghost(2)
        info = self.game.export_info(aid)
        self.assertGreater(info["bytes"], 0)
        self.assertTrue(os.path.exists(info["path"]))
        world_at_anchor = None  # recover the value the anchor froze
        from savepoint.state import load_wsave
        world_at_anchor = float(load_wsave(info["path"]).tensors["world"][0])

        # a "friend" receives the file and adopts it in a fresh game
        foreign = os.path.join(self.dir, "from_a_friend.wsave")
        shutil.copy2(info["path"], foreign)
        game2 = ButterflyGame(FakeHost(), Timeline(
            tempfile.mkdtemp(prefix="savepoint_test2_")), self.dir)
        game2.new_game(None, seed=5)
        imp = game2.import_save(foreign, label="from session 1")
        self.assertEqual(imp["type"], "imported")
        self.assertIn(imp["save_id"], imp["anchors"])
        rw = game2.rewind(imp["save_id"])
        self.assertEqual(rw["type"], "rewound")
        self.assertEqual(game2.host._world, world_at_anchor)

    def test_import_respects_slot_cap(self):
        for i in range(3):
            self.game.drop_anchor(f"a{i}")
        info = self.game.export_info(self.game.anchors[0])
        with self.assertRaises(RuntimeError):
            self.game.import_save(info["path"])


class MissionTestCase(unittest.TestCase):
    def setUp(self):
        import server.game as G
        self._orig = G.MISSION_EXPLORE_BLOCKS
        G.MISSION_EXPLORE_BLOCKS = 2       # short missions for tests
        self.G = G
        self.dir = tempfile.mkdtemp(prefix="savepoint_mission_")
        self.game = ButterflyGame(FakeHost(), Timeline(self.dir), self.dir)

    def tearDown(self):
        self.G.MISSION_EXPLORE_BLOCKS = self._orig
        shutil.rmtree(self.dir, ignore_errors=True)

    def _explore_and_tear(self):
        ms = self.game.start_mission(None, seed=3)
        self.assertEqual(ms["type"], "mission_started")
        self.assertIsNone(ms["anchor"])
        # first move drops the home anchor (streaming caches exist only now)
        _, _, s0 = self.game.step(dict(A_W))
        self.assertIn("home_anchor", s0)
        self.assertEqual(s0["mission"], {"phase": "explore", "tear_in": 2,
                                         "tear_now": False})
        _, _, s1 = self.game.step(dict(A_W))
        self.assertEqual(s1["mission"]["tear_in"], 1)
        _, _, s2 = self.game.step(dict(A_W))
        self.assertTrue(s2["mission"]["tear_now"])
        tear = self.game.mission_tear(seed=9)
        self.assertEqual(tear["type"], "mission_tear")
        self.assertEqual(tear["total"], 2)
        return tear

    def test_mission_held(self):
        self._explore_and_tear()
        # FakeHost is deterministic: mimicry is a perfect hold -> no strikes
        _, _, t1 = self.game.step(dict(A_W))
        self.assertEqual(t1["mission"]["strikes"], 0)
        _, _, end = self.game.step(dict(A_W))
        self.assertEqual(end["type"], "mission_end")
        self.assertTrue(end["won"])
        self.assertEqual(end["reason"], "held")
        self.assertIsNone(self.game.mission)
        self.assertEqual(self.game.mode, "free")

    def test_mission_torn_by_strikes(self):
        import server.game as G
        orig_strikes = G.MISSION_STRIKES
        G.MISSION_STRIKES = 2
        try:
            self._explore_and_tear()
            # a microscopic chaos baseline makes any divergence a strike
            self.game.chaos_curve = [1e-9] * 400
            _, _, t1 = self.game.step(dict(A_A))   # diverging action -> strike 1
            self.assertEqual(t1["mission"]["strikes"], 1)
            _, _, end = self.game.step(dict(A_A))  # strike 2 -> torn
            self.assertEqual(end["type"], "mission_end")
            self.assertFalse(end["won"])
            self.assertEqual(end["reason"], "torn")
            self.assertIsNone(self.game.mission)
            self.assertEqual(self.game.mode, "free")
        finally:
            G.MISSION_STRIKES = orig_strikes

    def test_mission_guards_controls(self):
        self.game.start_mission(None, seed=3)
        self.game.step(dict(A_W))          # home anchor drops here
        self.assertEqual(len(self.game.anchors), 1)
        for call in (lambda: self.game.drop_anchor("x"),
                     lambda: self.game.rewind(self.game.anchors[0]),
                     lambda: self.game.start_butterfly(self.game.anchors[0])):
            with self.assertRaises(RuntimeError):
                call()
        # abandoning frees the controls again
        self.game.abandon_mission()
        self.assertIsNone(self.game.mission)
        self.game.drop_anchor("now allowed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
