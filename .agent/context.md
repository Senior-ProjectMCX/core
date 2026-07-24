# DCS Two-Layer Phantom Simulation Handover Context

This file serves as a context bridge for future LLM sessions. It provides an immediate overview of the project's current state, findings, parameter values, and next steps.

---

## 1. Project Objective
Replicate and validate the diffuse correlation spectroscopy (DCS) simulation methodology from:
> **"Deep-learning-based separation of shallow and deep layer blood flow rates in diffuse correlation spectroscopy"** (Nakabayashi et al., 2023, *Biomedical Optics Express*).

Specifically: Simulate $g_2(\tau)$ autocorrelation curves for a two-layer dynamic flow phantom (shallow cutaneous layer and deep muscle layer) using Monte Carlo pathlengths and calibrate the physics engine to match the physical experimental dataset (`data/Dataset_phantom-intra40.mat`).

---

## 2. Project Directory Structure
```
d:\RA\
├── README.md                     # General setup and execution guide
├── .agent/
│   └── context.md                # This handover file
├── data/                         # MAT and CSV datasets
│   ├── Dataset_phantom-intra40.mat    # Physical validation dataset
│   ├── Dataset_simulated-intra40.mat  # Calibrated simulated dataset
│   ├── Dataset_metadata.csv           # Pump rates & estimated BFI (1540 x 7)
│   ├── Dataset_flattened.csv          # Metadata + flattened g2 curves (1540 x 1510)
│   └── pathlengths.npz                # High-stats 100M photon slab pathlengths
├── docs/                         # Papers and documentation
│   ├── boe-14-10-5358.pdf             # Reference scientific paper
│   ├── data.md                        # Variables definition for physical dataset
│   └── note.txt                       # User notes translated to academic English
├── plots/                        # Validation curve comparisons
│   ├── comparison_*.png               # Global dataset average curves
│   └── validation_*.png               # Target flow rates comparisons
├── survey/                       # Backups of unmodified original files & research
│   ├── simulation.py, dynamic_flow.py, generate_dataset.py, compare.py, lstm_model.py
│   └── dynamic_engine.py, calibrate_realistic.py, simulate_dcs.py, lstm_model.pth
└── code/                         # Active production scripts
    ├── simulation.py                  # Realistic geometry voxel MCX simulation
    ├── dynamic_flow.py                # g2 calculation with calibrated scale factors
    ├── generate_dataset.py            # Dataset generator loop (uses dynamic_flow)
    ├── compare.py                     # Evaluates curve differences (MAE/RMSE) and plots
    └── mat_to_csv.py                  # Utility to convert MATLAB datasets to CSV
```

---

## 3. Calibrated Physical Parameters
Through Nelder-Mead optimization on the slab geometry pathlength database, the following parameters were calibrated:
* **Fitted Coherence Factors ($\beta$):**
  * $1\text{ cm}$ SDS: **0.4051**
  * $2\text{ cm}$ SDS: **0.4943**
  * $3\text{ cm}$ SDS: **0.4845**
* **Optimized Dynamic Scaling Factors ($\text{scale}_s, \text{scale}_d$):**
  * Shallow layer flow scale ($\text{scale}_s$): **0.0962**
  * Deep layer flow scale ($\text{scale}_d$): **0.5492**
* **Translation Formula (in `code/dynamic_flow.py`):**
  $$D_{B, \text{shallow}} = 0.0962 \times FR_s \times 10^{-6}\text{ mm}^2/\text{s}$$
  $$D_{B, \text{deep}} = 0.5492 \times FR_d \times 10^{-6}\text{ mm}^2/\text{s}$$

---

## 4. Key Findings & Accomplishments
1. **Slab Model Dominance**: The **Slab Model** matches the physical dataset significantly better than the **Realistic Model**. In the realistic grid geometry ($0.5\text{ mm}$ resolution), modeling the thin $0.05\text{ mm}$ Mylar separator as a $0.5\text{ mm}$ voxel (10x too thick) acts as an artificial scattering block, distorting paths. The Slab Model (no separator) provides the correct physical boundary representation.
2. **Error Reduction**: Using the calibrated scaling factors, the simulated curves overlap closely with physical data, yielding a **22% reduction in Mean Absolute Error (MAE)** for the 1 cm SDS channel (dropped to **0.07122**).
3. **CSV Export**: The MAT dataset has been successfully flattened and exported to CSV (`data/Dataset_flattened.csv`), making it ready for tabular ML (e.g., XGBoost, Random Forest).
4. **LSTM Standby**: PyTorch LSTM trainer files have been moved into `survey/` to keep the active `code/` folder focused purely on the physical simulation.

---

## 5. Next Steps for the Next LLM
The user has approved the **Future Improvements Plan** (`docs/future_improvements_plan.md`). When resuming work:
1. **Implement Hybrid Convective-Brownian Model**: Modify `code/dynamic_flow.py` to support directed flow velocity ($V$):
   $$\langle \Delta r^2(\tau) \rangle = 6 D_B \tau + V^2 \tau^2$$
   Fit/calibrate $V$ to see if it reduces residual errors at late delay times.
2. **Grid Resolution Increase**: Test the MCX simulation at a higher resolution (e.g. `unitinmm = 0.1`) to represent the Mylar sheet at its physical thickness ($0.05\text{ mm}$ is close to 1 voxel at $0.1\text{ mm}$ or $0.05\text{ mm}$ resolution) to see if the realistic geometry matches or outperforms the slab model.
3. **Mie Scattering Phase Function**: Replace Henyey-Greenstein ($g=0.9$) with Mie theory parameters for the glass beads layer in MCX.
