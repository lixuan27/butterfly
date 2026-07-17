"""CPU tests for the one-seat/spectator-queue transport (no GPU, stub game)."""

import json
import os
import sys
import unittest

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJ, "src"))
os.environ["SAVEPOINT_NO_GPU"] = "1"   # startup skips model load


class StubTimeline:
    @staticmethod
    def to_tree():
        return []


class StubGame:
    timeline = StubTimeline()
    _current_anchor = None

    def step(self, action):
        return None, None, {"type": "collapsed", "mode": "idle",
                            "blocks_left": 0, "anchors": [], "cursor": {},
                            "current": None}


def drain_until(ws, want_type, max_msgs=20):
    for _ in range(max_msgs):
        m = ws.receive()
        if "text" in m:
            d = json.loads(m["text"])
            if d["type"] == want_type:
                return d
    raise AssertionError(f"no {want_type} within {max_msgs} messages")


class QueueTestCase(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import server.app as A
        A.CLIENTS.clear()
        A.PLAYER = None
        self.A = A
        self.client = TestClient(A.app)

    def test_no_gpu_graceful(self):
        self.A.GAME = None
        with self.client as c:
            with c.websocket_connect("/ws") as ws:
                drain_until(ws, "seat")
                ws.send_text(json.dumps({"type": "action", "keys": []}))
                err = drain_until(ws, "error")
                self.assertIn("no GPU", err["error"])

    def test_seat_and_spectator(self):
        self.A.GAME = StubGame()
        with self.client as c:
            with c.websocket_connect("/ws") as w1:
                s1 = drain_until(w1, "seat")
                self.assertEqual(s1["role"], "player")
                with c.websocket_connect("/ws") as w2:
                    s2 = drain_until(w2, "seat")
                    self.assertEqual(s2["role"], "waiting")
                    # spectator control is refused
                    w2.send_text(json.dumps({"type": "action", "keys": ["w"]}))
                    err = drain_until(w2, "error")
                    self.assertIn("seat is taken", err["error"])
                    # spectator read-only passthrough works
                    w2.send_text(json.dumps({"type": "tree"}))
                    self.assertEqual(drain_until(w2, "tree")["tree"], [])
                    # player acts; the event is broadcast to the spectator too
                    w1.send_text(json.dumps({"type": "action", "keys": ["w"]}))
                    self.assertEqual(
                        drain_until(w2, "collapsed")["type"], "collapsed")

    def test_promotion_on_disconnect(self):
        self.A.GAME = StubGame()
        with self.client as c:
            w1 = c.websocket_connect("/ws").__enter__()
            drain_until(w1, "seat")
            with c.websocket_connect("/ws") as w2:
                self.assertEqual(drain_until(w2, "seat")["role"], "waiting")
                w1.__exit__(None, None, None)      # the player leaves
                s = drain_until(w2, "seat")
                self.assertEqual(s["role"], "player")

    def test_health(self):
        self.A.GAME = None
        with self.client as c:
            h = c.get("/health").json()
            self.assertEqual(set(h), {"gpu", "clients", "seat_taken"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
