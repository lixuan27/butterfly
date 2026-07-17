"""End-to-end protocol test for the Butterfly game server (GPU required).

Runs the FastAPI app in-process via TestClient — no ports, no uvicorn — and
plays a full session: new dream -> free play -> anchor -> record ghost ->
duel vs chaos -> verdict -> rewind. Asserts message flow, frame delivery,
and prints latency stats for the realtime story.
"""

import json
import os
import sys
import time

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJ, "src"))


def recv_until_json(ws, want_types, collect_bins=None, max_msgs=400):
    """Consume messages until a JSON whose type is in want_types; count binaries."""
    for _ in range(max_msgs):
        m = ws.receive()
        if "text" in m:
            data = json.loads(m["text"])
            if data["type"] == "error":
                raise AssertionError(f"server error: {data['error']}")
            if data["type"] in want_types:
                return data
        elif "bytes" in m and collect_bins is not None:
            collect_bins.append(m["bytes"])
    raise AssertionError(f"no {want_types} within {max_msgs} messages")


def main():
    from fastapi.testclient import TestClient
    from server.app import app

    A = {"type": "action", "keys": ["w"], "mouse": [0.0, 0.05]}
    step_times = []

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # 1. new dream
            ws.send_text(json.dumps({"type": "new", "image": "0003.png", "seed": 7}))
            started = recv_until_json(ws, {"started"})
            assert started["mode"] == "free" and started["blocks_left"] == 120
            print(f"[e2e] started (prime {started['prime_ms']:.0f} ms)", flush=True)

            # 2. free play
            for _ in range(3):
                bins = []
                ws.send_text(json.dumps(A))
                t0 = time.time()
                info = recv_until_json(ws, {"stepped"}, bins)
                step_times.append((time.time() - t0) * 1000)
                assert len(bins) in (9, 12), f"unexpected frame count {len(bins)}"
            assert bins[0][0] == 0, "free frames must be live-stream (tag 0)"
            print(f"[e2e] free play ok ({len(bins)} frames/block)", flush=True)

            # 3. anchor + ghost recording
            ws.send_text(json.dumps({"type": "anchor", "label": "e2e"}))
            anch = recv_until_json(ws, {"anchored"})
            aid = anch["save_id"]
            assert anch["bytes"] > 2e9, "state anchor should be ~2.7GB"
            for _ in range(4):
                ws.send_text(json.dumps(A))
                recv_until_json(ws, {"stepped"}, [])
            print(f"[e2e] anchored {aid} + 4 ghost blocks", flush=True)

            # 4. duel
            ws.send_text(json.dumps({"type": "duel", "anchor_id": aid, "seed": 99}))
            ds = recv_until_json(ws, {"duel_started"})
            assert ds["total"] == 4 and len(ds["ghost_actions"]) == 4
            verdict, ghost_frames = None, 0
            for i in range(4):
                bins = []
                ws.send_text(json.dumps(dict(A)))  # try to mirror the ghost
                info = recv_until_json(ws, {"duel_tick", "duel_end"}, bins)
                ghost_frames += sum(1 for b in bins if b[0] == 1)
                live_frames = sum(1 for b in bins if b[0] == 0)
                assert live_frames > 0 and ghost_frames > 0, "duel must stream both"
                if info["type"] == "duel_end":
                    verdict = info
            assert verdict is not None, "duel must end after ghost exhausted"
            assert "grade" in verdict and len(verdict["dists"]) == 4
            print(f"[e2e] duel done: score={verdict['score']:.3f} "
                  f"grade={verdict['grade']} mean_dist={verdict['mean_dist']:.4f} "
                  f"baseline={verdict['chaos_baseline']:.4f}", flush=True)

            # 5. rewind + tree + inspect
            ws.send_text(json.dumps({"type": "rewind", "anchor_id": aid}))
            rw = recv_until_json(ws, {"rewound"})
            assert rw["mode"] == "free"
            ws.send_text(json.dumps({"type": "tree"}))
            tree = recv_until_json(ws, {"tree"})
            assert tree["tree"], "timeline tree must contain the anchor"
            ws.send_text(json.dumps({"type": "inspect"}))
            comp = recv_until_json(ws, {"inspect"})
            assert comp["components"]["__total__"] > 2e9
            print("[e2e] rewind/tree/inspect ok", flush=True)

    mean_ms = sum(step_times) / len(step_times)
    print(f"[e2e] step round-trip mean {mean_ms:.0f} ms "
          f"(min {min(step_times):.0f} / max {max(step_times):.0f})", flush=True)
    print("E2E PASS", flush=True)


if __name__ == "__main__":
    main()
