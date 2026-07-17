"""FastAPI + WebSocket transport for the Butterfly game (one GPU, one dream).

Protocol (client -> server, JSON):
    {"type":"new",    "image":"0003.png", "seed":1234}
    {"type":"action", "keys":["w"], "mouse":[0.0,0.1]}
    {"type":"anchor", "label":"before the bridge"}
    {"type":"rewind", "anchor_id":"..."}
    {"type":"duel",   "anchor_id":"...", "seed":7}
    {"type":"tree"} | {"type":"inspect"}

Server -> client: JSON events (started/stepped/anchored/duel_*/collapsed/...)
and binary frames: [1B stream: 0=live 1=ghost][4B LE frame idx][JPEG].
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB_DIR = os.path.join(PROJ, "web")

app = FastAPI(title="Butterfly — SavePoint")

GAME = None
SAVE_DIR = os.environ.get("SAVEPOINT_SAVE_DIR", os.path.join(PROJ, "saves"))
IMAGE_DIR = os.environ.get(
    "SAVEPOINT_IMAGE_DIR",
    os.path.join(PROJ, "third_party/mg2_src_fetch/Matrix-Game-2/demo_images/universal"))
LOCK = asyncio.Lock()


@app.on_event("startup")
def startup() -> None:
    global GAME
    if os.environ.get("SAVEPOINT_NO_GPU"):  # UI development without a model
        return
    import sys
    mg2_root = os.environ.get(
        "SAVEPOINT_MG2_ROOT",
        os.path.join(PROJ, "third_party/mg2_src_fetch/Matrix-Game-2"))
    sys.path.insert(0, mg2_root)
    sys.path.insert(0, os.path.join(PROJ, "src"))
    from savepoint.hosts.matrix_game2 import MatrixGame2Host
    from savepoint.timeline import Timeline
    from server.game import ButterflyGame
    host = MatrixGame2Host(
        mg2_root,
        os.environ.get("SAVEPOINT_CKPT_DIR", os.path.join(PROJ, "ckpts/Matrix-Game-2.0")),
        latents_window_frames=24)
    os.makedirs(SAVE_DIR, exist_ok=True)
    GAME = ButterflyGame(host, Timeline(SAVE_DIR), SAVE_DIR,
                         chaos_baseline_path=os.environ.get("SAVEPOINT_CHAOS_BASELINE"))
    # warm up torch.compile off the player's first move (one throwaway block)
    from server.game import ButterflyGame as _BG  # noqa: F401
    warm_img = _load_image(os.environ.get("SAVEPOINT_WARMUP_IMAGE", "0003.png"))
    host.prime(warm_img, seed=0)
    host.step({"keys": [], "mouse": [0.0, 0.0]})
    print("server warm: first-block compile done", flush=True)


@app.get("/")
def index() -> HTMLResponse:
    with open(os.path.join(WEB_DIR, "index.html")) as fh:
        return HTMLResponse(fh.read())


@app.get("/export/{anchor_id}")
def export(anchor_id: str) -> FileResponse:
    """Relay: download an anchor's .wsave (the file is immutable once written)."""
    from fastapi import HTTPException
    if GAME is None:
        raise HTTPException(503, "no world model loaded (SAVEPOINT_NO_GPU)")
    try:
        info = GAME.export_info(anchor_id)
    except KeyError:
        raise HTTPException(404, f"unknown anchor {anchor_id}")
    return FileResponse(info["path"], media_type="application/octet-stream",
                        filename=f"butterfly_{anchor_id[:8]}.wsave")


def _load_image(name: str):
    import sys
    sys.path.insert(0, os.path.join(PROJ, "src"))
    from bench.p0_smoke import load_start_image
    return load_start_image(os.path.join(IMAGE_DIR, os.path.basename(name)))


def _jpeg_frames(frames_u8, stream_id: int, first_index: int):
    """frames_u8: torch uint8 [T,H,W,3] -> length-prefixed binary messages."""
    out = []
    for i, frame in enumerate(frames_u8.numpy()):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(struct.pack("<BI", stream_id, first_index + i) + buf.tobytes())
    return out


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    frame_no = 0
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            mtype = msg.get("type")
            async with LOCK:
                try:
                    if mtype == "new":
                        image = _load_image(msg.get("image", "0003.png"))
                        t0 = time.time()
                        info = await asyncio.to_thread(
                            GAME.new_game, image, int(msg.get("seed", 1234)))
                        info["prime_ms"] = (time.time() - t0) * 1000
                        frame_no = 0
                        await sock.send_text(json.dumps(info))
                    elif mtype == "action":
                        t0 = time.time()
                        live, ghost, info = await asyncio.to_thread(
                            GAME.step, {"keys": msg.get("keys", []),
                                        "mouse": msg.get("mouse", [0.0, 0.0]),
                                        "flap": bool(msg.get("flap", False))})
                        if live is not None:
                            for blob in _jpeg_frames(live, 0, frame_no):
                                await sock.send_bytes(blob)
                            if ghost is not None:
                                for blob in _jpeg_frames(ghost, 1, frame_no):
                                    await sock.send_bytes(blob)
                            frame_no += live.shape[0]
                        info["step_ms"] = (time.time() - t0) * 1000
                        await sock.send_text(json.dumps(info))
                    elif mtype == "anchor":
                        await sock.send_text(json.dumps(await asyncio.to_thread(
                            GAME.drop_anchor, msg.get("label", ""))))
                    elif mtype == "rewind":
                        await sock.send_text(json.dumps(await asyncio.to_thread(
                            GAME.rewind, msg["anchor_id"])))
                    elif mtype == "duel":
                        await sock.send_text(json.dumps(await asyncio.to_thread(
                            GAME.start_duel, msg["anchor_id"],
                            int(msg.get("seed", int(time.time()) % 100000)))))
                    elif mtype == "butterfly":
                        await sock.send_text(json.dumps(await asyncio.to_thread(
                            GAME.start_butterfly, msg["anchor_id"])))
                    elif mtype == "import":
                        await sock.send_text(json.dumps(await asyncio.to_thread(
                            GAME.import_save, msg["path"], msg.get("label", ""))))
                    elif mtype == "tree":
                        await sock.send_text(json.dumps(
                            {"type": "tree", "tree": GAME.timeline.to_tree(),
                             "current": getattr(GAME, "_current_anchor", None)}))
                    elif mtype == "inspect":
                        comp = await asyncio.to_thread(GAME.host.state_components)
                        await sock.send_text(json.dumps(
                            {"type": "inspect",
                             "components": {k: int(v) for k, v in comp.items()}}))
                    else:
                        await sock.send_text(json.dumps(
                            {"type": "error", "error": f"unknown type {mtype}"}))
                except Exception as e:  # report to client; keep the socket alive
                    import traceback
                    traceback.print_exc()
                    await sock.send_text(json.dumps(
                        {"type": "error", "error": f"{type(e).__name__}: {e}"}))
    except WebSocketDisconnect:
        pass


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
