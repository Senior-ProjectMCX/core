# DCS Two-Layer Flow Phantom — Workflow (EN)

> Thai version: `workflow_th.md`. Results: `compare.md`.

## Papers
- **[P1] Nakabayashi et al., Biomed. Opt. Express 14(10), 5358 (2023)** — the two-layer phantom,
  the DCS setup, BFI estimation, and the c₂/c₁ test we validate against.
- **[P2] Wu et al., Neurophotonics 8(1), 015001 (2021)** — the momentum-transfer Monte Carlo
  forward model, Eq.(5), which is the core physics of our g₁ computation.

## Goal
Produce synthetic **g₂(r, τ) at r = 1, 2, 3 cm** from a one-time Monte Carlo photon run, and
compare it against the physical measured g₂. g₂ is the final deliverable; c₂/c₁ and BFI are
intermediate validation gates.

---

## What maps to which paper (the key cross-reference)

| Our code | What it computes | Paper source (exact) |
|---|---|---|
| `simulation.py` geometry | 2-layer slab: shallow 5.5 mm / deep 35 mm; μₐ=0.56, μ′ₛ=4.81 cm⁻¹ @785 nm; SDS 1/2/3 cm | **P1 §2.1–2.2** (phantom + instrumentation) |
| `simulation.py` MCX + momentum | photon-resolved pathlength L **and** momentum transfer Y per layer; 1e9 photons | **P2 §2.4.3** (MC-based fitting; "save path length and momentum transfer for each photon for each layer") |
| `dynamic_flow.py` `calculate_g1` | `G1(τ)=1/Np Σ exp(−⅓k₀²Σ Yᵢ⟨Δr²⟩ᵢ)·exp(−Σ μ_{a,i}Lᵢ)` | **P2 Eq.(5)** ← the core fix (replaces μ′ₛ·L) |
| `dynamic_flow.py` MSD term | `⟨Δr²(τ)⟩ᵢ = 6·αD_B,i·τ` (Brownian) | **P2 Eq.(3)** (= P1 uses the same 6·αD_B·τ inside Eq.3) |
| `dynamic_flow.py` `calculate_g2` | `g2 = 1 + β|g1|²` (Siegert), g1 = G1/I | **P2 Eq.(4)** = **P1 Eq.(2)** (identical Siegert) |
| `data_io.extract_beta` | β per SDS from physical g₂ at τ=0 | **P1 Eq.(2)** (β from the analytical fit / g2(0)−1) |
| `c2c1_test.py` BFI fit | semi-infinite `G2=1+β|(3μ′ₛ/4π)(e^{−Kr₁}/r₁−e^{−Kr₂}/r₂)/I|²`, `K²=3μₐμ′ₛ+6μ′ₛ²k₀²αD_Bτ` | **P1 Eq.(3)** ≡ **P2 Eq.(1)+(2)** (same semi-infinite Green's function) |
| `c2c1_test.py` regression | `BFI = c₁·FR_s + c₂·FR_d + c₃` (least squares) | **P1 Eq.(4)** |
| `c2c1_test.py` targets | c₂/c₁ = 0.14 / 0.38 / 0.46 | **P1 Eq.(10)–(12)** (0.113/0.804, 0.373/0.969, 0.624/1.359) |
| `compare.py` MAE/RMSE | curve error on held-out set | **P1 Eq.(5)–(9)** (metric definitions; P1 applies them to flow rates, we apply to g₂ curves) |
| `compare.py` held-out split | 70/30 train/test by flow combo | analogous to **P1 §2.4.2** (P1 used 90/10 for LSTM; we use 70/30 for calibration) |

Notes on differences:
- **Optical properties:** we use **P1 §2.1** phantom values (μₐ=0.56, μ′ₛ=4.81 cm⁻¹), NOT P2's
  forehead/calf values — because we are modeling P1's phantom.
- **Extrapolated boundary z_b:** P2 gives `z_b = 1.76/μ′ₛ`; P1 gives `z_b = 2(1−R_eff)/(3μ′ₛ(1+R_eff))`.
  We use P2's form in `c2c1_test.py`.
- **We do NOT use P1's LSTM (§2.4.2, Fig.2)** — that is P1's flow-separation method; our project
  validates the forward g₂ physics, not the inverse network.

---

## Pipeline

### Step 1 — `simulation.py` (MCX, run once) — [P2 §2.4.3, P1 §2.1–2.2]
GPU photon transport; saves partial pathlength **L** and momentum transfer **Y** per layer
(`ismomentum=1`, `savedetflag='dpm'`). Media columns: col0=shallow, col1=deep, col2=mylar,
col3=acrylic (0-indexed by medium label − 1, no background column).
```
python code/simulation.py --nphoton 1e9 --geometry simplified --output data/pathlengths.npz
```

### Step 2 — `dynamic_flow.py` + `generate_dataset.py` (g₂ synthesis) — [P2 Eq.(3)(4)(5)]
Eq.(5) with absorption weighting + normalization (g1(0)=1), Siegert to g₂, single calibration
`αD_B = a·FR` (same a both layers).
```
python code/generate_dataset.py --pathlengths data/pathlengths.npz --output data/Dataset_simulated.csv
```

### Step 3 — `c2c1_test.py` (GEOMETRY GATE) — [P1 Eq.(3)(4)(10–12)]
Fit BFI (semi-infinite), regress, report c₂/c₁ vs the same-fitter physical reference. Run this
**before** trusting curves; c₂/c₁ is invariant to the calibration scale a.
```
python code/c2c1_test.py --synthetic data/Dataset_simulated.csv
```

### Step 4 — `compare.py` (held-out metrics) — [P1 Eq.(5)–(9), §2.4.2]
70/30 split by combo, calibrate a on train, report MAE/RMSE on test only.
```
python code/compare.py --pathlengths data/pathlengths.npz --output-plot plots/g2_comparison.png
```

---

## Data source
Physical `.mat` is gone; `data_io.py` reads `data/metadata.csv` (FR set/meas + BFI, [P1] labels)
and `data/g2_SDS{1,2,3}cm.csv` (physical g₂). 1540 trials / 133 combos in the CSV (P1 text says
1340 / 134).

## What was fixed vs the old prototype
1. **[P2 Eq.5]** momentum transfer Y replaces the `μ′ₛ·L` diffusion approximation (the physics error).
2. Off-by-one bug: shallow/deep were read from the wrong ppath columns.
3. Single calibration factor (was two scales) so the **[P1]** c₂/c₁ test stays meaningful.
4. Held-out MAE/RMSE (was reported on fitted data).
