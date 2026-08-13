"""
Central data access for the DCS pipeline.

The original physical dataset (Dataset_phantom-intra40.mat) is no longer present; the
ground truth now lives in CSVs exported earlier:

    data/metadata.csv        : trial, FR_set/meas shallow/deep, BFI at 1/2/3 cm
    data/g2_SDS{1,2,3}cm.csv : physical g2(r, tau), one file per source-detector separation

All loaders read those CSVs so no .mat file is required anywhere in the pipeline.
"""
import os
import numpy as np
import pandas as pd

# Delay-time axis: tau_0us .. tau_500us  => 501 points, 1 us step.
N_TAU = 501
TAU_ARRAY = np.arange(N_TAU) * 1e-6          # seconds
SDS_LABELS = ['1cm', '2cm', '3cm']

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_metadata(data_dir=_DEFAULT_DATA_DIR):
    """
    Load metadata.csv. Returns a dict of 1D arrays (length = n_trials):
      FR_set_s, FR_set_d      : preset pump flow rates (ml/min/100g)
      FR_meas_s, FR_meas_d    : measured outlet flow rates (ground truth for regression)
      BFI                     : (3, n_trials) physical BFI at 1/2/3 cm (units 1e-9 cm^2/s)
      n_trials                : number of trials actually present
    """
    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    n_trials = len(df)
    return {
        'FR_set_s': df['FR_set_shallow_ml_min_100g'].to_numpy(float),
        'FR_set_d': df['FR_set_deep_ml_min_100g'].to_numpy(float),
        'FR_meas_s': df['FR_meas_shallow_ml_min_100g'].to_numpy(float),
        'FR_meas_d': df['FR_meas_deep_ml_min_100g'].to_numpy(float),
        'BFI': np.vstack([
            df['BFI_SDS1cm_1e-9cm2_s'].to_numpy(float),
            df['BFI_SDS2cm_1e-9cm2_s'].to_numpy(float),
            df['BFI_SDS3cm_1e-9cm2_s'].to_numpy(float),
        ]),
        'n_trials': n_trials,
    }


def load_physical_g2(data_dir=_DEFAULT_DATA_DIR):
    """
    Load the three g2_SDS*cm.csv files into a single array of shape (3, N_TAU, n_trials),
    matching the (sds, tau, trial) layout the rest of the pipeline expects.
    """
    curves = []
    for lbl in SDS_LABELS:
        df = pd.read_csv(os.path.join(data_dir, f'g2_SDS{lbl}.csv'))
        tau_cols = [c for c in df.columns if c.startswith('tau_')]
        assert len(tau_cols) == N_TAU, f'expected {N_TAU} tau columns, got {len(tau_cols)}'
        curves.append(df[tau_cols].to_numpy(float).T)   # (N_TAU, n_trials)
    return np.stack(curves, axis=0)                       # (3, N_TAU, n_trials)


def extract_beta(physical_g2):
    """
    Coherence factor per SDS from the physical g2 at tau=0: beta = mean_trials(g2(0)) - 1.
    Expected ~0.407 / 0.494 / 0.490 (Nakabayashi dataset). physical_g2 shape (3, N_TAU, n_trials).
    """
    return np.array([np.mean(physical_g2[s, 0, :]) - 1.0 for s in range(physical_g2.shape[0])])


def save_synthetic(path, sim_g2, meta):
    """
    Persist synthetic g2 in the same CSV layout as the physical data, so compare.py reads both
    through load_physical_g2. sim_g2 shape (3, N_TAU, n_trials).
    Writes <stem>_SDS{1,2,3}cm.csv next to `path`; returns the list of written files.
    """
    out_dir = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    tau_cols = [f'tau_{i}us' for i in range(N_TAU)]
    written = []
    for s, lbl in enumerate(SDS_LABELS):
        df = pd.DataFrame(sim_g2[s].T, columns=tau_cols)   # (n_trials, N_TAU)
        df.insert(0, 'trial', np.arange(sim_g2.shape[2]))
        fpath = os.path.join(out_dir, f'{stem}_SDS{lbl}.csv')
        df.to_csv(fpath, index=False)
        written.append(fpath)
    return written


def load_synthetic_g2(path, data_dir=None):
    """Load synthetic g2 written by save_synthetic; mirrors load_physical_g2 -> (3, N_TAU, n_trials)."""
    out_dir = data_dir or (os.path.dirname(os.path.abspath(path)) or '.')
    stem = os.path.splitext(os.path.basename(path))[0]
    curves = []
    for lbl in SDS_LABELS:
        df = pd.read_csv(os.path.join(out_dir, f'{stem}_SDS{lbl}.csv'))
        tau_cols = [c for c in df.columns if c.startswith('tau_')]
        curves.append(df[tau_cols].to_numpy(float).T)
    return np.stack(curves, axis=0)
