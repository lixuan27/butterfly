"""GPU e2e for the Butterfly mode + Relay import (in-process TestClient).

The scientific assertion inside a game test: with the saved RNG restored,
a faithful replay must be BIT-EXACT — so the pre-flap butterfly distance is
exactly 0.0 on real hardware, and any post-flap distance is caused by the one
changed action alone.
"""

import glob
import json
import os
import sys
import time

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJ, "src"))

from test_server_e2e import recv_until_json  # noqa: E402  (same dir)


def main():
    from fastapi.testclient import TestClient
    from server.app import app

    WALK = {"type": "action", "keys": ["w"], "mouse": [0.0, 0.05]}
    TURN = {"type": "action", "keys": ["s"], "mouse": [0.0, -0.1]}

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "new", "image": "0003.png", "seed": 7}))
            recv_until_json(ws, {"started"})
            for _ in range(2):
                ws.send_text(json.dumps(WALK))
                recv_until_json(ws, {"stepped"}, [])

            ws.send_text(json.dumps({"type": "anchor", "label": "bfly-e2e"}))
            aid = recv_until_json(ws, {"anchored"})["save_id"]
            for _ in range(3):
                ws.send_text(json.dumps(WALK))
                recv_until_json(ws, {"stepped"}, [])
            print(f"[bfly-e2e] anchored {aid} + 3 ghost blocks", flush=True)

            # -- butterfly ----------------------------------------------------
            ws.send_text(json.dumps({"type": "butterfly", "anchor_id": aid}))
            bs = recv_until_json(ws, {"butterfly_started"})
            assert bs["total"] == 3 and len(bs["ghost_actions"]) == 3

            # block 1: faithful replay -> BIT-EXACT (dist == 0.0), even on GPU
            bins = []
            ws.send_text(json.dumps(WALK))
            t1 = recv_until_json(ws, {"butterfly_tick"}, bins)
            assert not t1["flapped"] and t1["flap_block"] is None
            assert t1["dist"] == 0.0, \
                f"faithful replay must be bit-exact, got dist={t1['dist']}"
            assert sum(1 for b in bins if b[0] == 1) > 0, "ghost must stream"
            print(f"[bfly-e2e] pre-flap replay bit-exact (dist=0.0)", flush=True)

            # block 2: the one flap, different action -> divergence begins
            ws.send_text(json.dumps({**TURN, "flap": True}))
            t2 = recv_until_json(ws, {"butterfly_tick"}, [])
            assert t2["flapped"] and t2["flap_block"] == 1
            assert t2["dist"] > 0.0, "the flap must move the world"

            # block 3: replay resumes; episode ends with a verdict
            ws.send_text(json.dumps(WALK))
            end = recv_until_json(ws, {"butterfly_end"}, [])
            assert end["flap_block"] == 1 and len(end["dists"]) == 2
            assert end["amp"] > 0.0 and end["grade"] in "SABC"
            print(f"[bfly-e2e] flap amp={end['amp']:.2f}x chaos "
                  f"grade={end['grade']} dists={[round(d,4) for d in end['dists']]}",
                  flush=True)

            # -- relay: adopt our own .wsave as if a friend sent it -----------
            save_dir = os.environ.get("SAVEPOINT_SAVE_DIR",
                                      os.path.join(PROJ, "saves"))
            wsave = sorted(glob.glob(os.path.join(save_dir, "*.wsave")))[0]
            ws.send_text(json.dumps({"type": "import", "path": wsave,
                                     "label": "relayed"}))
            imp = recv_until_json(ws, {"imported"})
            assert imp["save_id"] in imp["anchors"]
            ws.send_text(json.dumps({"type": "rewind",
                                     "anchor_id": imp["save_id"]}))
            rw = recv_until_json(ws, {"rewound"})
            assert rw["mode"] == "free"
            ws.send_text(json.dumps(WALK))
            recv_until_json(ws, {"stepped"}, [])
            print(f"[bfly-e2e] relay import + rewind + step ok", flush=True)

    print("BUTTERFLY E2E PASS", flush=True)


def mission_flow():
    """Mission「守梦远征」 on GPU: explore -> tear -> hold -> verdict."""
    import server.game as G
    G.MISSION_EXPLORE_BLOCKS = 3           # short mission for the e2e
    from fastapi.testclient import TestClient
    from server.app import app

    WALK = {"type": "action", "keys": ["w"], "mouse": [0.0, 0.05]}
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "mission", "image": "0003.png",
                                     "seed": 11}))
            ms = recv_until_json(ws, {"mission_started"})
            assert ms["explore_blocks"] == 3 and ms["anchor"] is None
            tear_seen = home_seen = False
            for _ in range(4):        # move 1 drops home, then 3 explore moves
                ws.send_text(json.dumps(WALK))
                s = recv_until_json(ws, {"stepped"}, [])
                home_seen = home_seen or bool(s.get("home_anchor"))
                if s.get("mission", {}).get("tear_now"):
                    tear_seen = True
            assert home_seen, "home anchor must drop on the first move"
            assert tear_seen, "explore must end in tear_now"
            ws.send_text(json.dumps({"type": "mission_tear", "seed": 77}))
            tear = recv_until_json(ws, {"mission_tear"})
            assert tear["total"] == 3 and tear["strikes_allowed"] == 3
            end = None
            for _ in range(3):
                ws.send_text(json.dumps(WALK))     # mimic the ghost
                info = recv_until_json(ws, {"duel_tick", "mission_end"}, [])
                if info["type"] == "mission_end":
                    end = info
            assert end is not None and end["reason"] in ("held", "torn")
            print(f"[mission-e2e] {end['reason']} won={end['won']} "
                  f"strikes={end['strikes']} grade={end.get('grade')}", flush=True)
    print("MISSION E2E PASS", flush=True)


if __name__ == "__main__":
    main()
    mission_flow()
