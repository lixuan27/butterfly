# SaveBench: the three measurements

The save button doubles as an instrument. All protocols live in `src/bench/` and run on
one GPU; numbers below are from Matrix-Game-2.0 (1.8B distilled, 352×640) on 1×H200.

## M1 · Resume fidelity

Save → load → continue with identical actions and noise, versus an uninterrupted run.

- **Full-state `.wsave` (2,740 MB): bit-exact (PSNR ∞).** Deterministic replay holds on
  real hardware; the noise stream is a host-owned `torch.Generator` so global RNG pollution
  (e.g. from `torch.compile` autotune) cannot break it.
- Naïve per-tensor int8 on the caches (1,903 MB): **13.5 dB** — the world comes back wrong.
  Runtime state is far more quantization-fragile than weights.

## M2 · Fork divergence (the butterfly effect)

From one save, run pairs of continuations; measure latent MSE over 192 frames.

- Same actions, re-rolled noise → **chaos half-life T½ = 109.5 frames ≈ 4.4 s**.
- Same noise, one changed action → divergence rises **3.55×** faster
  (controllability SNR = action AUC / noise AUC = 31.9 / 9.0).
- These two curves are the game's difficulty system: Duel scores you against the measured
  chaos curve (mimicking your ghost's inputs scores ≈ 0 by construction — verified in the
  GPU e2e), and One Flap grades your single action's ripple against a full noise re-roll.

## M3 · Minimal sufficient state

How small can a save get before the world stops coming back?

| save | size | first-block fidelity |
|---|---|---|
| full state | 2,740 MB | ∞ |
| int8 (naïve) | 1,903 MB | ~41 dB, decays |
| re-prime, 24 latent frames | **4.8 MB** | **41.8 dB** |
| re-prime, 12 | 3.5 MB | 36.2 dB |
| re-prime, 6 | 2.8 MB | 26.3 dB |
| re-prime, 3 | 2.5 MB | 24.4 dB |

A 4.8 MB re-prime save matches the int8 full-state at **1/400th** the size — but decays
over horizon (30.8 → 13 dB), which is direct evidence that the cache carries memory
beyond the visible latent window. The rate-distortion curve of "how many tokens is a
world worth" has not saturated at 24 frames.

## Model physics used as game rules

| constant | value | game rule |
|---|---|---|
| attention window | 6 latent frames ≈ 0.85 s | the dream's short memory |
| RoPE budget | 360 blocks ≈ 164 s | dream lifetime, then collapse |
| block latency | ~0.5 s per world-move | "the dream has inertia" — steering, not twitch |
| chaos T½ | 4.4 s | the opponent |
| controllability SNR | 3.55 | why playing works at all |
