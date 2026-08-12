"""
Compare generated MC-DCS g2 curves with
the physical DCS reference dataset.

Core V1:
    shallow flow = 5
    deep flow = 20
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reference_loader import load_reference_condition
from dcs_processor import calculate_g1, calculate_g2


# ============================================================
# PATHS
# ============================================================

DETP_FILE = (
    Path(__file__).resolve().parent
    / "output"
    / "detp_core_v1.npy"
)


# ============================================================
# ERROR METRICS
# ============================================================

def calculate_error_metrics(
    generated_g2,
    reference_g2,
):
    generated_g2 = np.asarray(
        generated_g2,
        dtype=float,
    )

    reference_g2 = np.asarray(
        reference_g2,
        dtype=float,
    )

    if generated_g2.shape != reference_g2.shape:
        raise ValueError(
            "Generated and reference curves must have "
            f"the same shape, got "
            f"{generated_g2.shape} and "
            f"{reference_g2.shape}"
        )

    difference = generated_g2 - reference_g2

    mae = np.mean(
        np.abs(difference)
    )

    rmse = np.sqrt(
        np.mean(difference ** 2)
    )

    max_abs_error = np.max(
        np.abs(difference)
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
    }


# ============================================================
# LOAD PHOTON HISTORIES
# ============================================================

def load_detp():

    if not DETP_FILE.exists():
        raise FileNotFoundError(
            f"Detected-photon file not found:\n{DETP_FILE}"
        )

    detp = np.load(
        DETP_FILE
    )

    if detp.ndim != 2:
        raise ValueError(
            f"Expected 2D detp array, got {detp.shape}"
        )

    if detp.shape[0] != 5:
        raise ValueError(
            "Expected DPM photon histories with "
            f"5 rows, got {detp.shape[0]}"
        )

    return detp


# ============================================================
# LOAD PHYSICAL REFERENCE
# ============================================================

def load_core_reference():

    return load_reference_condition(
        shallow_flow=5,
        deep_flow=20,
    )


# ============================================================
# REFERENCE CURVES
# ============================================================

def get_reference_g2(reference):

    reference_g2 = {}

    for sds_cm in (1, 2, 3):

        reference_g2[sds_cm] = np.asarray(
            reference["g2"][sds_cm]["mean"],
            dtype=float,
        )

    return reference_g2


# ============================================================
# GENERATE g1 / g2
# ============================================================

def generate_curves(
    detp,
    reference_g2,
):

    tau_us = np.arange(
        501,
        dtype=float,
    )

    tau_seconds = (
        tau_us
        * 1e-6
    )

    generated_g1 = {}
    generated_g2 = {}
    beta = {}

    for sds_cm in (1, 2, 3):

        # Detector IDs correspond to SDS 1, 2, 3 cm
        detector_id = sds_cm

        g1 = calculate_g1(
            detp=detp,
            tau_seconds=tau_seconds,
            detector_id=detector_id,
        )

        # Effective experimental coherence factor
        # from the physical reference intercept.
        beta_value = (
            reference_g2[sds_cm][0]
            - 1.0
        )

        g2 = calculate_g2(
            g1=g1,
            beta=beta_value,
        )

        generated_g1[sds_cm] = g1
        generated_g2[sds_cm] = g2
        beta[sds_cm] = beta_value

    return (
        tau_us,
        generated_g1,
        generated_g2,
        beta,
    )


# ============================================================
# PLOT
# ============================================================

def plot_comparison(
    tau_us,
    generated_g2,
    reference_g2,
):

    for sds_cm in (1, 2, 3):

        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(
            tau_us,
            reference_g2[sds_cm],
            label="Physical Reference",
        )

        plt.plot(
            tau_us,
            generated_g2[sds_cm],
            label="Generated",
        )

        plt.xlabel(
            "Delay time τ (µs)"
        )

        plt.ylabel(
            "g₂(τ)"
        )

        plt.title(
            "Generated vs Physical Reference\n"
            f"SDS {sds_cm} cm"
        )

        plt.xlim(
            0,
            500,
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== Core V1: Generated vs Physical Reference ==="
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    detp = load_detp()

    reference = load_core_reference()

    reference_g2 = get_reference_g2(
        reference
    )

    print(
        f"\nDetected photons: "
        f"{detp.shape[1]}"
    )

    print(
        "Reference condition: "
        f"shallow={reference['commanded_shallow']}, "
        f"deep={reference['commanded_deep']}"
    )

    print(
        f"Reference trials: "
        f"{len(reference['trial_indices'])}"
    )

    # --------------------------------------------------------
    # Generate curves
    # --------------------------------------------------------

    (
        tau_us,
        generated_g1,
        generated_g2,
        beta,
    ) = generate_curves(
        detp=detp,
        reference_g2=reference_g2,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    print(
        "\n=== Comparison Results ==="
    )

    for sds_cm in (1, 2, 3):

        metrics = calculate_error_metrics(
            generated_g2[sds_cm],
            reference_g2[sds_cm],
        )

        print(
            f"\nSDS {sds_cm} cm"
        )

        print(
            f"beta: "
            f"{beta[sds_cm]:.6f}"
        )

        print(
            f"Generated g2(0): "
            f"{generated_g2[sds_cm][0]:.6f}"
        )

        print(
            f"Reference g2(0): "
            f"{reference_g2[sds_cm][0]:.6f}"
        )

        print(
            f"Generated g2(100 us): "
            f"{generated_g2[sds_cm][100]:.6f}"
        )

        print(
            f"Reference g2(100 us): "
            f"{reference_g2[sds_cm][100]:.6f}"
        )

        print(
            f"Generated g2(500 us): "
            f"{generated_g2[sds_cm][-1]:.6f}"
        )

        print(
            f"Reference g2(500 us): "
            f"{reference_g2[sds_cm][-1]:.6f}"
        )

        print(
            f"MAE: "
            f"{metrics['mae']:.6f}"
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.6f}"
        )

        print(
            f"Max absolute error: "
            f"{metrics['max_abs_error']:.6f}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_comparison(
        tau_us=tau_us,
        generated_g2=generated_g2,
        reference_g2=reference_g2,
    )


if __name__ == "__main__":
    main()