"""CPU tests for SaveBench metric primitives."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bench.metrics import (  # noqa: E402
    per_frame_mse, per_frame_psnr, feature_distance, pairwise_divergence,
    half_life, saturation_level, controllability_snr,
)


def test_per_frame_metrics():
    a = torch.zeros(4, 3, 8, 8)
    b = a.clone()
    b[2] += 0.5
    mse = per_frame_mse(a, b)
    assert mse[0] == 0 and abs(mse[2].item() - 0.25) < 1e-6
    psnr = per_frame_psnr(a, b)
    assert torch.isinf(psnr[0]) and abs(psnr[2].item() - 6.0206) < 1e-3
    print("PASS per_frame_metrics")


def test_feature_distance():
    fa = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    fb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = feature_distance(fa, fb)
    assert abs(d[0].item()) < 1e-6 and abs(d[1].item() - 1.0) < 1e-6
    print("PASS feature_distance")


def test_half_life_interpolation():
    curve = torch.tensor([0.0, 0.2, 0.6, 0.9])
    # crosses 0.4 midway between t=1 (0.2) and t=2 (0.6) -> 1.5
    assert abs(half_life(curve, 0.4) - 1.5) < 1e-6
    assert half_life(curve, 2.0) is None
    assert half_life(curve, 0.0) == 0.0
    # dt scaling (e.g. frames -> seconds at 25 fps)
    assert abs(half_life(curve, 0.4, dt=0.04) - 0.06) < 1e-6
    print("PASS half_life")


def test_saturation_and_snr():
    t = torch.arange(100).float()
    sat = 1 - torch.exp(-t / 10)         # saturating divergence
    plateau = saturation_level(sat)
    assert 0.98 < plateau <= 1.0
    thalf = half_life(sat, plateau / 2)
    assert 6.0 < thalf < 8.0              # analytic: 10*ln(2)≈6.93
    noise = sat * 0.25
    snr = controllability_snr(sat, noise)
    assert abs(snr["snr"] - 4.0) < 1e-5 and snr["horizon"] == 100
    print("PASS saturation_and_snr")


def test_pairwise_divergence():
    base = torch.zeros(6, 3, 4, 4)
    r1, r2, r3 = base.clone(), base.clone(), base.clone()
    r2[3:] += 1.0
    r3[3:] -= 1.0
    curve = pairwise_divergence([r1, r2, r3])
    assert torch.all(curve[:3] == 0)
    # pairs: (r1,r2)=1, (r1,r3)=1, (r2,r3)=4 -> mean 2.0 after t=3
    assert abs(curve[4].item() - 2.0) < 1e-6
    print("PASS pairwise_divergence")


if __name__ == "__main__":
    test_per_frame_metrics()
    test_feature_distance()
    test_half_life_interpolation()
    test_saturation_and_snr()
    test_pairwise_divergence()
    print("ALL 5 TESTS PASS")
