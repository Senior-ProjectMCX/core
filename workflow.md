# DCS Two-Layer Flow Phantom — Workflow Summary

## Overview
Simulate and validate **Diffuse Correlation Spectroscopy (DCS)** g2(t) autocorrelation curves for a two-layer tissue phantom, then compare against a physical dataset (1540 trials, 3 SDS channels).

---

## Pipeline (3 Steps)

### Step 1 — `simulation.py` · MCX Monte Carlo
**What it does:** Builds a 3D voxel grid of the phantom and fires 100M photons through it via GPU (`pmcx`). Records **partial pathlengths** (L?, L_d) of detected photons per layer.

**Why this method:** Monte Carlo is the gold-standard for modelling photon transport in scattering tissue. GPU acceleration (MCX) makes 100M photons tractable. Partial pathlengths are the minimal statistic needed to compute DCS — no full trajectory storage required.

```
Input : geometry type (slab / realistic), nphoton
Output: pathlengths.npz  ?  {det_1_Ls, det_1_Ld, det_2_Ls, …}
```

---

### Step 2 — `dynamic_flow.py` + `generate_dataset.py` · g2 Synthesis
**What it does:**
1. Converts flow rates (ml/min) ? Brownian diffusion coefficients D_B using **calibrated scaling factors** (scale_s = 0.0962, scale_d = 0.5492).
2. Computes g1(t) = ?exp(-G? t)? by averaging over all detected photons.
3. Applies Siegert relation: **g2(t) = 1 + ß |g1(t)|²**.
4. Loops over all 1540 experimental trials and saves a `.mat` file.

**Why this method:**
- The **photon-averaging approach** uses the actual Monte Carlo pathlength distribution, making it physically exact (no analytical slab approximation).
- ß (coherence factor) is fitted directly from the physical data at t?0 (g2(0) - 1), so it absorbs real optical alignment effects.
- Downsampling to 10k photons per detector keeps runtime tractable with negligible statistical error.

```
Input : pathlengths.npz, Dataset_phantom-intra40.mat (flow rates + ß fitting)
Output: Dataset_simulated-intra40.mat
```

---

### Step 3 — `compare.py` · Quantitative Validation
**What it does:** Loads both physical and simulated datasets, computes **MAE & RMSE** across all 1540 × 501 curve points for each SDS, and plots four representative flow combinations.

**Why this method:** MAE and RMSE together capture both average and large-deviation errors, giving a balanced picture of simulation quality.

```
Input : Dataset_phantom-intra40.mat, Dataset_simulated-intra40.mat
Output: g2_comparison.png + printed metrics
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Two-layer **slab** geometry (not realistic) | Realistic model over-thickens the 0.05 mm Mylar sheet to 0.5 mm (voxel limit), creating a spurious high-scatter barrier |
| GPU MCX with 100M photons | Statistically stable pathlength distributions; runtime < minutes on RTX 4050 |
| Calibrated D_B scaling (Nelder-Mead) | Bridges the gap between volumetric pump flow and effective Brownian diffusion in the intralipid phantom |
| ß fitted per SDS from physical data | Accounts for spatial coherence loss and real detector optics at each separation |

---

## Quick-Run Commands

```powershell
# Activate environment
.\venv_mcx\Scripts\Activate.ps1

# Step 1 - Monte Carlo simulation
python .\code\simulation.py --nphoton 1e8 --geometry simplified --output .\data\pathlengths.npz

# Step 2 - Generate synthetic dataset
python .\code\generate_dataset.py --pathlengths .\data\pathlengths.npz --physical-mat .\data\Dataset_phantom-intra40.mat --output .\data\Dataset_simulated-intra40.mat

# Step 3 - Compare & plot
python .\code\compare.py --physical .\data\Dataset_phantom-intra40.mat --simulated .\data\Dataset_simulated-intra40.mat --output-plot .\plots\g2_comparison.png
```
