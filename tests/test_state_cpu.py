"""CPU-only unit tests for savepoint.state — no GPU, no model weights needed.

The fake caches mirror Matrix-Game 2.0's real runtime-state topology
(4 named cache groups + host-owned VAE conv cache), at miniature sizes.

Run: python tests/test_state_cpu.py
"""

import os
import sys
import tempfile
import zipfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savepoint.state import (  # noqa: E402
    WorldState, save_wsave, load_wsave, snapshot_rng, restore_rng,
    _HEADER_NAME, _TENSORS_NAME, _ACTIONS_NAME, _RNG_NAME,
)

BLOCKS = 4  # miniature stand-in for MG2's 30 transformer blocks


def make_mg2_shaped_caches(seed: int):
    """Same group names and per-block key structure as Matrix-Game 2.0."""
    g = torch.Generator().manual_seed(seed)

    def kv_group(seq, heads, dim, batch=1):
        return [{
            "k": torch.randn(batch, seq, heads, dim, generator=g),
            "v": torch.randn(batch, seq, heads, dim, generator=g),
            "global_end_index": torch.tensor([37], dtype=torch.long),
            "local_end_index": torch.tensor([37], dtype=torch.long),
        } for _ in range(BLOCKS)]

    caches = {
        "kv": kv_group(seq=96, heads=2, dim=16),
        "kv_mouse": kv_group(seq=5, heads=2, dim=8, batch=6),   # B*frame_seq_length rows
        "kv_keyboard": kv_group(seq=5, heads=2, dim=8),
        "cross": [{
            "k": torch.randn(1, 9, 2, 16, generator=g),
            "v": torch.randn(1, 9, 2, 16, generator=g),
            "is_init": True,
        } for _ in range(BLOCKS)],
    }
    vae_cache = {f"vae_cache.{i}": torch.randn(1, 3, 4, 4, generator=g) for i in range(3)}
    return caches, vae_cache


def _capture(caches, tensors, **kw):
    return WorldState.capture(
        caches, tensors, cursor={"latent_frame": 9},
        actions=[{"t": 0, "keys": ["w"], "mouse": [0.1, -0.2]}],
        meta={"model_id": "fake/mg2-mini", "resolution": [352, 640]}, **kw)


def test_roundtrip_bitexact():
    caches, vae = make_mg2_shaped_caches(seed=1)
    state = _capture(caches, vae)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wsave")
        save_wsave(state, p)
        loaded = load_wsave(p)
    targets, _ = make_mg2_shaped_caches(seed=2)  # different content, same shapes
    loaded.restore_caches(targets, restore_rng_state=False)
    for group in caches:
        for a, b in zip(caches[group], targets[group]):
            for key, val in a.items():
                if isinstance(val, torch.Tensor):
                    assert torch.equal(val, b[key]), f"{group}.{key} not bit-exact"
                else:
                    assert b[key] == val
    for name, t in vae.items():
        assert torch.equal(loaded.tensors[name], t), f"{name} not bit-exact"
    assert loaded.cursor["latent_frame"] == 9
    assert loaded.actions[0]["keys"] == ["w"]
    print("PASS roundtrip_bitexact")


def test_int8_quant_bounded_error():
    caches, vae = make_mg2_shaped_caches(seed=3)
    state = _capture(caches, vae)
    with tempfile.TemporaryDirectory() as d:
        p8, praw = os.path.join(d, "q8.wsave"), os.path.join(d, "raw.wsave")
        save_wsave(state, p8, quant="int8")
        save_wsave(state, praw)
        ratio = os.path.getsize(p8) / os.path.getsize(praw)
        loaded = load_wsave(p8)
    ref = caches["kv"][0]["k"]
    err = (loaded.caches["kv"][0]["k"] - ref).abs().max().item()
    assert err <= ref.abs().max().item() / 127.0 + 1e-6, f"int8 error {err} exceeds bound"
    assert ratio < 0.45, f"int8 file not smaller: ratio={ratio:.2f}"
    # cursors and non-cache tensors stay exact under quantization
    assert torch.equal(loaded.caches["kv"][0]["global_end_index"],
                       caches["kv"][0]["global_end_index"])
    for name, t in vae.items():
        assert torch.equal(loaded.tensors[name], t)
    print(f"PASS int8_quant (max_err={err:.4g}, size_ratio={ratio:.2f})")


def test_rng_restore_reproduces_sequence():
    torch.manual_seed(1234)
    torch.randn(7)  # advance the stream
    snap = snapshot_rng()
    a = torch.randn(5)
    restore_rng(snap)
    b = torch.randn(5)
    assert torch.equal(a, b), "torch CPU RNG not reproduced"
    print("PASS rng_restore")


def test_reprime_mode_small_file():
    caches, _ = make_mg2_shaped_caches(seed=4)
    lat = torch.randn(1, 5, 4, 8, 8)
    state = _capture(None, {"latents": lat}, mode="reprime")
    assert not state.caches
    with tempfile.TemporaryDirectory() as d:
        pr, pf = os.path.join(d, "r.wsave"), os.path.join(d, "f.wsave")
        save_wsave(state, pr)
        full = _capture(caches, {"latents": lat})
        save_wsave(full, pf)
        assert os.path.getsize(pr) < 0.2 * os.path.getsize(pf)
        loaded = load_wsave(pr)
    assert torch.equal(loaded.tensors["latents"], lat)
    try:
        loaded.restore_caches(caches)
        raise AssertionError("reprime restore_caches should refuse")
    except ValueError:
        pass
    print("PASS reprime_mode")


def test_container_is_pickle_free():
    caches, vae = make_mg2_shaped_caches(seed=6)
    state = _capture(caches, vae)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wsave")
        save_wsave(state, p)
        with zipfile.ZipFile(p) as zf:
            names = set(zf.namelist())
    assert names == {_HEADER_NAME, _TENSORS_NAME, _ACTIONS_NAME, _RNG_NAME}, names
    print("PASS container_members")


def test_mismatches_rejected():
    caches, vae = make_mg2_shaped_caches(seed=7)
    state = _capture(caches, vae)
    # shape mismatch
    bad, _ = make_mg2_shaped_caches(seed=8)
    bad["kv"][0]["k"] = torch.randn(1, 97, 2, 16)
    try:
        state.restore_caches(bad, restore_rng_state=False)
        raise AssertionError("shape mismatch should be rejected")
    except ValueError:
        pass
    # missing cache group
    partial, _ = make_mg2_shaped_caches(seed=9)
    del partial["kv_mouse"]
    try:
        state.restore_caches(partial, restore_rng_state=False)
        raise AssertionError("missing group should be rejected")
    except KeyError:
        pass
    print("PASS mismatches_rejected")


def test_nbytes_accounting():
    caches, vae = make_mg2_shaped_caches(seed=10)
    state = _capture(caches, vae)
    rep = state.nbytes()
    assert set(rep) == {"cache.kv", "cache.kv_mouse", "cache.kv_keyboard",
                        "cache.cross", "tensor.vae_cache.0", "tensor.vae_cache.1",
                        "tensor.vae_cache.2", "__total__"}
    assert rep["__total__"] == sum(v for k, v in rep.items() if k != "__total__")
    assert rep["cache.kv"] > rep["cache.kv_keyboard"]
    print("PASS nbytes_accounting")


if __name__ == "__main__":
    test_roundtrip_bitexact()
    test_int8_quant_bounded_error()
    test_rng_restore_reproduces_sequence()
    test_reprime_mode_small_file()
    test_container_is_pickle_free()
    test_mismatches_rejected()
    test_nbytes_accounting()
    print("ALL 7 TESTS PASS")
