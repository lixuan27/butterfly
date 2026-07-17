"""P0 feasibility gate for SavePoint on Matrix-Game 2.0 (single GPU).

Measures, in one process:
  1. throughput      — fps / block latency / VRAM (gate: fps >= 12)
  2. determinism     — same seed, two episodes: are latents bit-identical?
  3. M1 resume       — three tracks from one branch point:
       A  uninterrupted reference
       B  state-save  -> .wsave -> restore -> continue
       C  reprime-save -> .wsave -> restore -> continue
     gate: track B bit-exact (or eps < perceptual threshold); track C measured
  4. save-file sizes — state raw / state int8 / reprime

Writes results.json + .wsave artifacts + per-track PSNR curves to --out_dir.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.metrics import per_frame_mse, per_frame_psnr  # noqa: E402
from savepoint.state import save_wsave, load_wsave  # noqa: E402


def scripted_actions(n_blocks):
    """Deterministic, human-plausible action script: walk forward, look around,
    strafe. Fixed regardless of seed so every track replays the same inputs."""
    acts = []
    for i in range(n_blocks):
        phase = i % 8
        if phase < 3:
            acts.append({"keys": ["w"], "mouse": [0.0, 0.0]})
        elif phase < 5:
            acts.append({"keys": ["w"], "mouse": [0.0, 0.1]})
        elif phase < 6:
            acts.append({"keys": ["a"], "mouse": [0.0, -0.1]})
        else:
            acts.append({"keys": ["d"], "mouse": [0.05, 0.0]})
    return acts


def load_start_image(path):
    from PIL import Image
    from torchvision.transforms import v2
    img = Image.open(path).convert("RGB")
    w, h = img.size
    th, tw = 352, 640
    if h / w > th / tw:
        nw, nh = w, int(w * th / tw)
    else:
        nh, nw = h, int(h * tw / th)
    img = img.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2))
    proc = v2.Compose([v2.Resize(size=(th, tw), antialias=True), v2.ToTensor(),
                       v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    return proc(img)[None, :, None]  # [1,3,1,352,640]


def rollout(host, image, seed, actions):
    host.prime(image, seed=seed)
    lat, pix, stats = [], [], []
    for a in actions:
        r = host.step(a)
        lat.append(r.latents.float().cpu())
        pix.append(r.frames.float().cpu())
        stats.append(r.stats)
    return torch.cat(lat, dim=1), torch.cat(pix, dim=0), stats


def continue_rollout(host, actions):
    lat, pix = [], []
    for a in actions:
        r = host.step(a)
        lat.append(r.latents.float().cpu())
        pix.append(r.frames.float().cpu())
    return torch.cat(lat, dim=1), torch.cat(pix, dim=0)


def compare(tag, lat_ref, pix_ref, lat_alt, pix_alt, results):
    bit = torch.equal(lat_ref, lat_alt)
    lat_mse = per_frame_mse(lat_ref.transpose(0, 1), lat_alt.transpose(0, 1))
    psnr = per_frame_psnr(pix_ref, pix_alt)
    results[tag] = {
        "latents_bit_exact": bool(bit),
        "latent_mse_per_frame": [float(x) for x in lat_mse],
        "pixel_psnr_per_frame": [float(x) for x in psnr],
        "pixel_psnr_min": float(psnr.min()),
    }
    print(f"[{tag}] bit_exact={bit} min_psnr={psnr.min():.2f} "
          f"max_latent_mse={lat_mse.max():.3e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mg2_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--n_blocks", type=int, default=24)
    ap.add_argument("--n_resume_blocks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.mg2_root)
    from savepoint.hosts.matrix_game2 import MatrixGame2Host  # noqa: E402

    os.makedirs(args.out_dir, exist_ok=True)
    image_path = args.image or os.path.join(
        args.mg2_root, "demo_images/universal/0000.png")
    image = load_start_image(image_path)
    results = {"config": vars(args), "image": image_path}

    host = MatrixGame2Host(args.mg2_root, args.ckpt_dir,
                           compile_vae=not args.no_compile)
    print("host ready", flush=True)

    n1 = args.n_blocks - args.n_resume_blocks
    acts = scripted_actions(args.n_blocks)

    # ---- 1) throughput + reference episode -------------------------------
    torch.cuda.reset_peak_memory_stats()
    lat_ref, pix_ref, stats = rollout(host, image, args.seed, acts)
    warm = stats[2:]  # skip compile/warmup blocks
    block_ms = sum(s["block_ms"] for s in warm) / len(warm)
    fps = (pix_ref.shape[0] / args.n_blocks) / (block_ms / 1000)
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    results["throughput"] = {"block_ms_mean": block_ms, "fps": fps,
                             "vram_gb": vram_gb,
                             "first_block_ms": stats[0]["block_ms"]}
    print(f"[throughput] {fps:.1f} fps ({block_ms:.0f} ms/block, "
          f"VRAM {vram_gb:.1f} GB)", flush=True)

    # ---- 2) determinism: full identical rerun ----------------------------
    lat_rep, pix_rep, _ = rollout(host, image, args.seed, acts)
    compare("determinism_rerun", lat_ref, pix_ref, lat_rep, pix_rep, results)

    # ---- 3) M1 three tracks ----------------------------------------------
    # branch-point episode: run n1 blocks, capture both save modes
    host.prime(image, seed=args.seed)
    for a in acts[:n1]:
        host.step(a)
    ws_state = host.capture(mode="state")
    ws_reprime = host.capture(mode="reprime")
    sizes = {}
    p_state = os.path.join(args.out_dir, "branch_state.wsave")
    p_state8 = os.path.join(args.out_dir, "branch_state_int8.wsave")
    p_rep = os.path.join(args.out_dir, "branch_reprime.wsave")
    save_wsave(ws_state, p_state)
    save_wsave(ws_state, p_state8, quant="int8")
    save_wsave(ws_reprime, p_rep)
    for name, p in [("state", p_state), ("state_int8", p_state8), ("reprime", p_rep)]:
        sizes[name] = os.path.getsize(p)
        print(f"[wsave] {name}: {sizes[name]/1e6:.1f} MB", flush=True)
    results["wsave_bytes"] = sizes
    results["state_components"] = {k: int(v) for k, v in ws_state.nbytes().items()}

    # track A reference tail: continue the branch episode uninterrupted
    lat_tail_ref, pix_tail_ref = continue_rollout(host, acts[n1:])

    # track B: load state-save from disk, restore, continue
    ws_b = load_wsave(p_state)
    host.restore(ws_b)
    lat_tail_b, pix_tail_b = continue_rollout(host, acts[n1:])
    compare("m1_state_resume", lat_tail_ref, pix_tail_ref, lat_tail_b, pix_tail_b, results)

    # track B8: int8-quantized state save
    ws_b8 = load_wsave(p_state8)
    host.restore(ws_b8)
    lat_b8, pix_b8 = continue_rollout(host, acts[n1:])
    compare("m1_state_int8_resume", lat_tail_ref, pix_tail_ref, lat_b8, pix_b8, results)

    # track C: reprime-save
    ws_c = load_wsave(p_rep)
    host.restore(ws_c)
    lat_tail_c, pix_tail_c = continue_rollout(host, acts[n1:])
    compare("m1_reprime_resume", lat_tail_ref, pix_tail_ref, lat_tail_c, pix_tail_c, results)

    # ---- 4) gate verdict ---------------------------------------------------
    gate_fps = fps >= 12
    bit_exact = results["m1_state_resume"]["latents_bit_exact"]
    perceptual = results["m1_state_resume"]["pixel_psnr_min"] > 35
    if gate_fps and bit_exact:
        verdict = "GO"
    elif gate_fps and perceptual:
        verdict = "GO_EPSILON"  # resumes, but not bit-exact — investigate before P1
    else:
        verdict = "NO-GO"
    results["gate"] = {"fps_ok": bool(gate_fps),
                       "state_resume_bit_exact": bool(bit_exact),
                       "state_resume_perceptual_ok": bool(perceptual),
                       "determinism_rerun_bit_exact":
                           results["determinism_rerun"]["latents_bit_exact"],
                       "verdict": verdict}
    print(f"[GATE] fps_ok={gate_fps} bit_exact={bit_exact} "
          f"perceptual={perceptual} => {verdict}", flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("results written to", os.path.join(args.out_dir, "results.json"), flush=True)


if __name__ == "__main__":
    main()
