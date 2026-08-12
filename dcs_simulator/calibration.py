"""
Calibration of effective alpha*Db parameters for DCS Simulator Core V1.

Fits:
    alphaDb_shallow
    alphaDb_deep

against all three physical-reference g2 curves simultaneously.

Important:
These are fitted effective parameters.
They are NOT assumed to be a direct conversion from commanded flow.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from parameters import (
    WAVELENGTH_NM,
    REFRACTIVE_INDEX_N,
    MU_A_MM,
)

from reference_loader import load_reference_condition
from dcs_processor import extract_photon_data


# ============================================================
# PATHS
# ============================================================

DETP_FILE = (
    Path(__file__).resolve().parent
    / "output"
    / "detp_core_v1.npy"
)


# ============================================================
# CONSTANTS
# ============================================================

WAVELENGTH_MM = WAVELENGTH_NM * 1e-6

K0 = (
    2.0
    * np.pi
    * REFRACTIVE_INDEX_N
    / WAVELENGTH_MM
)

TAU_US = np.arange(
    501,
    dtype=float,
)

TAU_SECONDS = (
    TAU_US
    * 1e-6
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DETP_FILE.exists():
        raise FileNotFoundError(
            f"Detected-photon file not found:\n{DETP_FILE}"
        )

    detp = np.load(DETP_FILE)

    reference = load_reference_condition(
        shallow_flow=5,
        deep_flow=20,
    )

    return detp, reference


# ============================================================
# PREPARE PHOTON DATA
# ============================================================

def prepare_detector_data(detp):

    photon_data = extract_photon_data(detp)

    detector_data = {}

    for detector_id in (1, 2, 3):

        mask = (
            photon_data["detector_id"]
            == detector_id
        )

        if not np.any(mask):
            raise ValueError(
                f"No photons detected at detector {detector_id}"
            )

        path_shallow = (
            photon_data["path_shallow"][mask]
        )

        path_deep = (
            photon_data["path_deep"][mask]
        )

        momentum_shallow = (
            photon_data["momentum_shallow"][mask]
        )

        momentum_deep = (
            photon_data["momentum_deep"][mask]
        )

        absorption_weight = np.exp(
            -MU_A_MM
            * (
                path_shallow
                + path_deep
            )
        )

        detector_data[detector_id] = {
            "momentum_shallow": momentum_shallow,
            "momentum_deep": momentum_deep,
            "weight": absorption_weight,
            "weight_sum": np.sum(absorption_weight),
        }

    return detector_data


# ============================================================
# GENERATE g1
# ============================================================

def calculate_g1_with_parameters(
    detector,
    alpha_db_shallow,
    alpha_db_deep,
):

    decay_rate = (
        2.0
        * K0**2
        * (
            detector["momentum_shallow"]
            * alpha_db_shallow
            +
            detector["momentum_deep"]
            * alpha_db_deep
        )
    )

    # Shape:
    # photons x tau
    dynamic_term = np.exp(
        -decay_rate[:, None]
        * TAU_SECONDS[None, :]
    )

    weighted_sum = np.sum(
        detector["weight"][:, None]
        * dynamic_term,
        axis=0,
    )

    g1 = (
        weighted_sum
        / detector["weight_sum"]
    )

    g1 = (
        g1
        / g1[0]
    )

    return g1


# ============================================================
# GENERATE g2
# ============================================================

def calculate_generated_g2(
    detector_data,
    reference,
    alpha_db_shallow,
    alpha_db_deep,
):

    generated = {}

    for sds_cm in (1, 2, 3):

        g1 = calculate_g1_with_parameters(
            detector=detector_data[sds_cm],
            alpha_db_shallow=alpha_db_shallow,
            alpha_db_deep=alpha_db_deep,
        )

        reference_g2 = (
            reference["g2"][sds_cm]["mean"]
        )

        beta = (
            reference_g2[0]
            - 1.0
        )

        generated[sds_cm] = (
            1.0
            + beta
            * np.abs(g1) ** 2
        )

    return generated


# ============================================================
# OPTIMIZATION OBJECTIVE
# ============================================================

def residuals(
    log_parameters,
    detector_data,
    reference,
):
    """
    Optimize in log10 space so alpha*Db always stays positive.

    log_parameters[0] = log10(alphaDb_shallow)
    log_parameters[1] = log10(alphaDb_deep)
    """

    alpha_db_shallow = (
        10.0 ** log_parameters[0]
    )

    alpha_db_deep = (
        10.0 ** log_parameters[1]
    )

    generated = calculate_generated_g2(
        detector_data=detector_data,
        reference=reference,
        alpha_db_shallow=alpha_db_shallow,
        alpha_db_deep=alpha_db_deep,
    )

    all_residuals = []

    for sds_cm in (1, 2, 3):

        reference_g2 = (
            reference["g2"][sds_cm]["mean"]
        )

        difference = (
            generated[sds_cm]
            - reference_g2
        )

        all_residuals.append(
            difference
        )

    return np.concatenate(
        all_residuals
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    generated,
    reference,
):

    metrics = {}

    all_errors = []

    for sds_cm in (1, 2, 3):

        reference_g2 = (
            reference["g2"][sds_cm]["mean"]
        )

        error = (
            generated[sds_cm]
            - reference_g2
        )

        metrics[sds_cm] = {
            "mae": np.mean(
                np.abs(error)
            ),
            "rmse": np.sqrt(
                np.mean(error ** 2)
            ),
            "max_error": np.max(
                np.abs(error)
            ),
        }

        all_errors.append(
            error
        )

    all_errors = np.concatenate(
        all_errors
    )

    combined_rmse = np.sqrt(
        np.mean(
            all_errors ** 2
        )
    )

    return metrics, combined_rmse


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== DCS Effective Parameter Calibration ==="
    )

    detp, reference = load_data()

    detector_data = prepare_detector_data(
        detp
    )

    print(
        f"Detected photons: {detp.shape[1]}"
    )

    for detector_id in (1, 2, 3):

        count = len(
            detector_data[detector_id][
                "momentum_shallow"
            ]
        )

        print(
            f"SDS {detector_id} cm: "
            f"{count} photons"
        )

    # --------------------------------------------------------
    # Initial assumption used by Core V1
    # --------------------------------------------------------

    initial_shallow = 5e-6
    initial_deep = 20e-6

    initial_log = np.log10(
        [
            initial_shallow,
            initial_deep,
        ]
    )

    print(
        "\nInitial parameters:"
    )

    print(
        f"alphaDb shallow = "
        f"{initial_shallow:.6e} mm^2/s"
    )

    print(
        f"alphaDb deep    = "
        f"{initial_deep:.6e} mm^2/s"
    )

    # --------------------------------------------------------
    # Initial error
    # --------------------------------------------------------

    initial_generated = (
        calculate_generated_g2(
            detector_data,
            reference,
            initial_shallow,
            initial_deep,
        )
    )

    _, initial_combined_rmse = (
        calculate_metrics(
            initial_generated,
            reference,
        )
    )

    print(
        f"Initial combined RMSE = "
        f"{initial_combined_rmse:.6f}"
    )

    # --------------------------------------------------------
    # Fit
    #
    # Search range:
    # 1e-8 to 1e-3 mm^2/s
    # --------------------------------------------------------

    result = least_squares(
        residuals,
        x0=initial_log,
        bounds=(
            [-8.0, -8.0],
            [-3.0, -3.0],
        ),
        args=(
            detector_data,
            reference,
        ),
        verbose=0,
    )

    best_shallow = (
        10.0 ** result.x[0]
    )

    best_deep = (
        10.0 ** result.x[1]
    )

    # --------------------------------------------------------
    # Final curves / metrics
    # --------------------------------------------------------

    generated = (
        calculate_generated_g2(
            detector_data,
            reference,
            best_shallow,
            best_deep,
        )
    )

    metrics, combined_rmse = (
        calculate_metrics(
            generated,
            reference,
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n=== Best-fit Effective Parameters ==="
    )

    print(
        f"alphaDb shallow = "
        f"{best_shallow:.8e} mm^2/s"
    )

    print(
        f"alphaDb deep    = "
        f"{best_deep:.8e} mm^2/s"
    )

    print(
        "\n=== Fit Quality ==="
    )

    print(
        f"Initial combined RMSE: "
        f"{initial_combined_rmse:.6f}"
    )

    print(
        f"Final combined RMSE:   "
        f"{combined_rmse:.6f}"
    )

    improvement = (
        100.0
        * (
            initial_combined_rmse
            - combined_rmse
        )
        / initial_combined_rmse
    )

    print(
        f"RMSE improvement:      "
        f"{improvement:.2f}%"
    )

    for sds_cm in (1, 2, 3):

        print(
            f"\nSDS {sds_cm} cm"
        )

        print(
            f"MAE: "
            f"{metrics[sds_cm]['mae']:.6f}"
        )

        print(
            f"RMSE: "
            f"{metrics[sds_cm]['rmse']:.6f}"
        )

        print(
            f"Max absolute error: "
            f"{metrics[sds_cm]['max_error']:.6f}"
        )

        print(
            f"Generated g2(100 us): "
            f"{generated[sds_cm][100]:.6f}"
        )

        print(
            f"Reference g2(100 us): "
            f"{reference['g2'][sds_cm]['mean'][100]:.6f}"
        )

    print(
        "\nOptimization success:",
        result.success,
    )

    print(
        "Optimizer message:",
        result.message,
    )


if __name__ == "__main__":
    main()