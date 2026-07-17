"""Record the SavePoint showcase videos (launch assets, no UI needed).

Three clips, all driven through the real save/restore path:
  1. showcase.mp4       — play, SAVE, keep playing, then LOAD: the world snaps back
  2. fork_2up.mp4       — one save, two futures (different noise), side by side
  3. actions_2up.mp4    — one save, two *choices* (turn left vs right), side by side
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from savepoint.state import load_wsave, save_wsave  # noqa: E402


def to_uint8(frames):
    return (frames.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)


def write_mp4(path, frames_uint8, fps=25):
    import imageio
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames_uint8:
        w.append_data(f)
    w.close()
    print(f"[mp4] {path} ({len(frames_uint8)} frames)", flush=True)


def banner(frames_uint8, text, n_frames=20):
    """Overlay a label strip on the first n frames (cv2)."""
    import cv2
    out = frames_uint8.copy()
    for i in range(min(n_frames, len(out))):
        cv2.rectangle(out[i], (0, 0), (out[i].shape[1], 34), (0, 0, 0), -1)
        cv2.putText(out[i], text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def rollout(host, actions):
    pix = []
    for a in actions:
        pix.append(host.step(a).frames.float().cpu())
    return torch.cat(pix, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mg2_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    sys.path.insert(0, args.mg2_root)
    from savepoint.hosts.matrix_game2 import MatrixGame2Host  # noqa: E402
    from bench.p0_smoke import load_start_image  # noqa: E402

    os.makedirs(args.out_dir, exist_ok=True)
    image = load_start_image(args.image or os.path.join(
        args.mg2_root, "demo_images/universal/0003.png"))
    host = MatrixGame2Host(args.mg2_root, args.ckpt_dir, latents_window_frames=24)

    F = {"keys": ["w"], "mouse": [0.0, 0.0]}
    L = {"keys": ["w"], "mouse": [0.0, -0.1]}
    R = {"keys": ["w"], "mouse": [0.0, 0.1]}

    # ---- clip 1: save / load ------------------------------------------------
    host.prime(image, seed=args.seed)
    seg1 = rollout(host, [F] * 6 + [R] * 2)               # approach
    save_path = os.path.join(args.out_dir, "showcase.wsave")
    save_wsave(host.capture(mode="state"), save_path)
    seg2 = rollout(host, [L] * 3 + [F] * 5)               # wander off
    host.restore(load_wsave(save_path))                    # snap back
    seg3 = rollout(host, [R] * 3 + [F] * 5)               # a different life
    clip = np.concatenate([
        banner(to_uint8(seg1), "playing...", 999),
        banner(to_uint8(seg2), "wandered off  (SAVED 8 blocks ago)", 999),
        banner(to_uint8(seg3), "LOADED the save -> same world, new choice", 999),
    ])
    write_mp4(os.path.join(args.out_dir, "showcase.mp4"), clip)

    # ---- clip 2: noise fork, 2-up ------------------------------------------
    state = load_wsave(save_path)
    host.restore(state); host._rng.manual_seed(1)
    a1 = to_uint8(rollout(host, [F] * 10))
    host.restore(state); host._rng.manual_seed(2)
    a2 = to_uint8(rollout(host, [F] * 10))
    two_up = np.concatenate([a1, a2], axis=2)  # side by side
    write_mp4(os.path.join(args.out_dir, "fork_2up.mp4"),
              banner(two_up, "one save, two futures (same actions, different chance)", 999))

    # ---- clip 3: action fork, 2-up -----------------------------------------
    host.restore(state)
    b1 = to_uint8(rollout(host, [L] * 10))
    host.restore(state)
    b2 = to_uint8(rollout(host, [R] * 10))
    write_mp4(os.path.join(args.out_dir, "actions_2up.mp4"),
              banner(np.concatenate([b1, b2], axis=2),
                     "one save, two choices (turn left vs right)", 999))

    print("showcase assets done", flush=True)


if __name__ == "__main__":
    main()
