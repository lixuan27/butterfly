"""M3 preview: reprime-window sweep — how many latent frames buy back the world?

One episode, one branch point, reprime saves at window sizes {3,6,12,24} plus
the full state save as reference. For each save: restore, replay the same
action tail, record per-frame PSNR vs the uninterrupted run. Output is the
first rate-distortion curve of SaveBench M3: save size (MB) vs resume fidelity
(initial PSNR = reconstruction gap; tail PSNR = after chaos amplification).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.metrics import per_frame_psnr  # noqa: E402
from savepoint.state import load_wsave, save_wsave  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mg2_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--warmup_blocks", type=int, default=10)   # 30 latent frames
    ap.add_argument("--tail_blocks", type=int, default=12)
    ap.add_argument("--windows", type=int, nargs="+", default=[3, 6, 12, 24])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sys.path.insert(0, args.mg2_root)
    from savepoint.hosts.matrix_game2 import MatrixGame2Host  # noqa: E402
    from bench.p0_smoke import load_start_image, scripted_actions  # noqa: E402

    os.makedirs(args.out_dir, exist_ok=True)
    image = load_start_image(args.image or os.path.join(
        args.mg2_root, "demo_images/universal/0000.png"))
    host = MatrixGame2Host(args.mg2_root, args.ckpt_dir,
                           latents_window_frames=max(args.windows))

    acts = scripted_actions(args.warmup_blocks + args.tail_blocks)
    warm, tail = acts[:args.warmup_blocks], acts[args.warmup_blocks:]

    # branch-point episode + saves
    host.prime(image, seed=args.seed)
    for a in warm:
        host.step(a)
    saves = {}
    p_state = os.path.join(args.out_dir, "sweep_state.wsave")
    save_wsave(host.capture(mode="state"), p_state)
    saves["state"] = p_state
    for w in args.windows:
        p = os.path.join(args.out_dir, f"sweep_reprime_w{w:02d}.wsave")
        save_wsave(host.capture(mode="reprime", reprime_window=w), p)
        saves[f"reprime_w{w}"] = p
        print(f"[save] w={w}: {os.path.getsize(p)/1e6:.1f} MB", flush=True)

    # uninterrupted reference tail
    ref = []
    for a in tail:
        ref.append(host.step(a).frames.float().cpu())
    ref = torch.cat(ref, dim=0)

    results = {"config": vars(args), "tracks": {}}
    for name, path in saves.items():
        host.restore(load_wsave(path))
        pix = []
        for a in tail:
            pix.append(host.step(a).frames.float().cpu())
        pix = torch.cat(pix, dim=0)
        psnr = per_frame_psnr(ref, pix)
        results["tracks"][name] = {
            "wsave_mb": os.path.getsize(path) / 1e6,
            "psnr_per_frame": [float(x) for x in psnr],
            "psnr_first_block": float(psnr[:12].mean()),
            "psnr_last_block": float(psnr[-12:].mean()),
            "bit_exact": bool(torch.equal(ref, pix)),
        }
        print(f"[{name}] {results['tracks'][name]['wsave_mb']:.1f} MB  "
              f"first={results['tracks'][name]['psnr_first_block']:.1f} dB  "
              f"last={results['tracks'][name]['psnr_last_block']:.1f} dB", flush=True)

    with open(os.path.join(args.out_dir, "m3_sweep_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("results written", flush=True)


if __name__ == "__main__":
    main()
