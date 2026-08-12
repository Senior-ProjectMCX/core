"""
Baseline DCS model used only for benchmark comparison.

This module reproduces the pathlength-based dynamic-flow formulation
from the supplied baseline project. It is intentionally kept separate
from the proposed model.
"""

from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PATHLENGTH_FILE = BASE_DIR / "baseline_data" / "pathlengths.npz"


def load_baseline_pathlengths():
    if not PATHLENGTH_FILE.exists():
        raise FileNotFoundError(
            "Baseline pathlength file not found:\n"
            f"{PATHLENGTH_FILE}"
        )

    data = np.load(PATHLENGTH_FILE)

    result = {}
    for detector_id in (1, 2, 3):
        result[detector_id] = (
            data[f"det_{detector_id}_Ls"],
            data[f"det_{detector_id}_Ld"],
        )

    return result


def calculate_baseline_g1(
    Ls,
    Ld,
    flow_s,
    flow_d,
    tau_array,
):
    if len(Ls) == 0:
        return np.ones_like(tau_array, dtype=float)

    lambda_vac = 785e-6
    n_tissue = 1.37
    k0 = 2 * np.pi * n_tissue / lambda_vac
    mus_p = 0.481

    # Calibrated factors from the supplied baseline implementation.
    DB_s = 0.0962 * float(flow_s) * 1e-6
    DB_d = 0.5492 * float(flow_d) * 1e-6

    coeff = 2.0 * (k0 ** 2) * mus_p
    decay_rates = coeff * (
        DB_s * np.asarray(Ls)
        + DB_d * np.asarray(Ld)
    )

    g1 = np.zeros_like(tau_array, dtype=float)

    for index, tau in enumerate(tau_array):
        g1[index] = np.mean(
            np.exp(-decay_rates * tau)
        )

    return g1


def calculate_baseline_g2(
    Ls,
    Ld,
    flow_s,
    flow_d,
    tau_array,
    beta,
):
    g1 = calculate_baseline_g1(
        Ls=Ls,
        Ld=Ld,
        flow_s=flow_s,
        flow_d=flow_d,
        tau_array=tau_array,
    )

    return 1.0 + float(beta) * (g1 ** 2)
