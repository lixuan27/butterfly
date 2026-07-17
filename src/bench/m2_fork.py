"""M2: fork divergence — the butterfly effect of a neural game, measured.

From one branch-point save, grow N futures under (a) identical actions but
different noise streams ("noise forks") and (b) identical noise seeds but
different actions ("action forks"). Divergence curves over time give:

  * T-half   — how fast the world forgets it was the same world
  * controllability SNR — do actions steer the world above the chaos floor?

This is the H1 measurement instrument: run it on the distilled model and on
the undistilled teacher, compare T-half.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.metrics import (  # noqa: E402
    controllability_snr, half_life, pairwise_divergence, per_frame_mse,
    saturation_level)
from savepoint.state import load_wsave, save_wsave  # noqa: E402

ACTION_MENU = {
    "forward": {"keys": ["w"], "mouse": [0.0, 0.0]},
    "left":    {"keys": ["w"], "mouse": [0.0, -0.1]},
    "right":   {"keys": ["w"], "mouse": [0.0, 0.1]},
    "strafe":  {"keys": ["a"], "mouse": [0.0, 0.0]},
}


def tail_rollout(host, actions):
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
    ap.add_argument("--warmup_blocks", type=int, default=8)
    ap.add_argument("--fork_blocks", type=int, default=16)
    ap.add_argument("--n_noise_forks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sys.path.insert(0, args.mg2_root)
    from savepoint.hosts.matrix_game2 import MatrixGame2Host  # noqa: E402
    from bench.p0_smoke import load_start_image, scripted_actions  # noqa: E402

    os.makedirs(args.out_dir, exist_ok=True)
    image = load_start_image(args.image or os.path.join(
        args.mg2_root, "demo_images/universal/0000.png"))
    host = MatrixGame2Host(args.mg2_root, args.ckpt_dir)

    # branch point
    host.prime(image, seed=args.seed)
    for a in scripted_actions(args.warmup_blocks):
        host.step(a)
    branch = os.path.join(args.out_dir, "branch.wsave")
    save_wsave(host.capture(mode="state"), branch)
    state = load_wsave(branch)
    results = {"config": vars(args)}

    # --- noise forks: same actions, different futures --------------------
    same_actions = [ACTION_MENU["forward"]] * args.fork_blocks
    noise_rollouts = []
    for i in range(args.n_noise_forks):
        host.restore(state)
        host._rng.manual_seed(args.seed * 7919 + i)
        noise_rollouts.append(tail_rollout(host, same_actions))
        print(f"[noise fork {i}] done", flush=True)
    noise_curve = pairwise_divergence(noise_rollouts, per_frame_mse)

    # --- action forks: same noise, different choices ----------------------
    action_rollouts = []
    for name in ("forward", "left", "right", "strafe"):
        host.restore(state)  # restores the saved RNG stream => same noise
        action_rollouts.append(tail_rollout(host, [ACTION_MENU[name]] * args.fork_blocks))
        print(f"[action fork {name}] done", flush=True)
    action_curve = pairwise_divergence(action_rollouts, per_frame_mse)

    plateau = saturation_level(noise_curve)
    results["noise"] = {
        "curve": [float(x) for x in noise_curve],
        "plateau": plateau,
        "t_half_frames": half_life(noise_curve, plateau / 2),
    }
    results["action"] = {"curve": [float(x) for x in action_curve]}
    results["controllability_snr"] = controllability_snr(action_curve, noise_curve)
    print(f"[M2] noise T1/2 = {results['noise']['t_half_frames']} frames; "
          f"SNR = {results['controllability_snr']['snr']:.2f}", flush=True)

    with open(os.path.join(args.out_dir, "m2_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    torch.save({"noise_rollouts": [r.half() for r in noise_rollouts],
                "action_rollouts": [r.half() for r in action_rollouts]},
               os.path.join(args.out_dir, "m2_rollouts.pt"))
    print("written", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
