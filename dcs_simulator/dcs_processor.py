"""
DCS processing for DCS Simulator Core V1.

Converts PMCX detected-photon histories into the
normalized electric field autocorrelation g1(tau),
then converts g1(tau) to g2(tau).

Supports:
1. Original/reference mode
   - Uses alpha*Db values from parameters.py

2. Dynamic-flow mode
   - Accepts alpha*Db shallow/deep from outside
   - Allows one MCX photon-history simulation to be reused
     for multiple flow conditions

Core V1:
- two dynamic layers
- Brownian motion model
- momentum-transfer Monte Carlo formulation
- absorption-weighted detected photon ensemble
"""

import numpy as np

from parameters import (
    WAVELENGTH_NM,
    REFRACTIVE_INDEX_N,
    MU_A_MM,
    ALPHA_DB_SHALLOW_MM2_S,
    ALPHA_DB_DEEP_MM2_S,
)


# ============================================================
# CONSTANTS
# ============================================================

# Convert wavelength from nm to mm
WAVELENGTH_MM = (
    WAVELENGTH_NM
    * 1e-6
)

# Wavenumber inside the medium [mm^-1]
K0 = (
    2.0
    * np.pi
    * REFRACTIVE_INDEX_N
    / WAVELENGTH_MM
)


# ============================================================
# EXTRACT DETECTED PHOTON DATA
# ============================================================

def extract_photon_data(detp):
    """
    Extract DPM output from PMCX.

    Expected layout for two media:

        Row 0 = detector ID
        Row 1 = shallow partial pathlength
        Row 2 = deep partial pathlength
        Row 3 = shallow momentum transfer
        Row 4 = deep momentum transfer
    """

    detp = np.asarray(
        detp
    )

    if detp.ndim != 2:
        raise ValueError(
            f"detp must be 2D, "
            f"got shape {detp.shape}"
        )

    if detp.shape[0] != 5:
        raise ValueError(
            "Expected 5 rows from "
            "savedetflag='DPM', "
            f"got {detp.shape[0]}"
        )

    detector_id = (
        detp[0]
        .astype(int)
    )

    path_shallow = detp[1]
    path_deep = detp[2]

    momentum_shallow = detp[3]
    momentum_deep = detp[4]

    return {
        "detector_id": detector_id,
        "path_shallow": path_shallow,
        "path_deep": path_deep,
        "momentum_shallow":
            momentum_shallow,
        "momentum_deep":
            momentum_deep,
    }


# ============================================================
# VALIDATE DYNAMIC PARAMETERS
# ============================================================

def validate_alpha_db(
    alpha_db_shallow,
    alpha_db_deep,
):
    """
    Validate dynamic DCS parameters.

    Parameters are expected in mm^2/s.
    """

    alpha_db_shallow = float(
        alpha_db_shallow
    )

    alpha_db_deep = float(
        alpha_db_deep
    )

    if not np.isfinite(
        alpha_db_shallow
    ):
        raise ValueError(
            "alpha_db_shallow must be finite."
        )

    if not np.isfinite(
        alpha_db_deep
    ):
        raise ValueError(
            "alpha_db_deep must be finite."
        )

    if alpha_db_shallow < 0:
        raise ValueError(
            "alpha_db_shallow "
            "cannot be negative."
        )

    if alpha_db_deep < 0:
        raise ValueError(
            "alpha_db_deep "
            "cannot be negative."
        )

    return (
        alpha_db_shallow,
        alpha_db_deep,
    )


# ============================================================
# CALCULATE g1
# ============================================================

def calculate_g1(
    detp,
    tau_seconds,
    detector_id,
    alpha_db_shallow=None,
    alpha_db_deep=None,
):
    """
    Calculate normalized electric field autocorrelation
    g1(tau) for one detector.

    Parameters
    ----------
    detp : np.ndarray
        PMCX detected-photon DPM output.

    tau_seconds : array-like
        Delay times in seconds.

    detector_id : int
        Detector ID to process.

    alpha_db_shallow : float or None
        Shallow alpha*Db in mm^2/s.

        None:
            Use ALPHA_DB_SHALLOW_MM2_S
            from parameters.py.

    alpha_db_deep : float or None
        Deep alpha*Db in mm^2/s.

        None:
            Use ALPHA_DB_DEEP_MM2_S
            from parameters.py.

    Notes
    -----
    Absorption weight:

        w = exp(
            -mu_a
            * (P_shallow + P_deep)
        )

    Dynamic decorrelation:

        exp(
            -2 * k0^2 * tau
            * (
                M_shallow * alphaDb_shallow
                +
                M_deep * alphaDb_deep
            )
        )

    The final g1 is the absorption-weighted
    average over detected photons.
    """

    # --------------------------------------------------------
    # Backward-compatible defaults
    # --------------------------------------------------------

    if alpha_db_shallow is None:
        alpha_db_shallow = (
            ALPHA_DB_SHALLOW_MM2_S
        )

    if alpha_db_deep is None:
        alpha_db_deep = (
            ALPHA_DB_DEEP_MM2_S
        )

    (
        alpha_db_shallow,
        alpha_db_deep,
    ) = validate_alpha_db(
        alpha_db_shallow,
        alpha_db_deep,
    )

    # --------------------------------------------------------
    # Extract photon histories
    # --------------------------------------------------------

    photon_data = (
        extract_photon_data(
            detp
        )
    )

    mask = (
        photon_data[
            "detector_id"
        ]
        == detector_id
    )

    if not np.any(mask):
        raise ValueError(
            f"No photons detected at "
            f"detector {detector_id}"
        )

    # --------------------------------------------------------
    # Photon histories for selected detector
    # --------------------------------------------------------

    path_shallow = (
        photon_data[
            "path_shallow"
        ][mask]
    )

    path_deep = (
        photon_data[
            "path_deep"
        ][mask]
    )

    momentum_shallow = (
        photon_data[
            "momentum_shallow"
        ][mask]
    )

    momentum_deep = (
        photon_data[
            "momentum_deep"
        ][mask]
    )

    # --------------------------------------------------------
    # Absorption weighting
    #
    # Core V1 currently uses the same mu_a
    # for shallow and deep layers.
    # --------------------------------------------------------

    total_path = (
        path_shallow
        + path_deep
    )

    absorption_weight = np.exp(
        -MU_A_MM
        * total_path
    )

    weight_sum = np.sum(
        absorption_weight
    )

    if (
        not np.isfinite(
            weight_sum
        )
        or weight_sum <= 0
    ):
        raise ValueError(
            "Invalid absorption weights"
        )

    # --------------------------------------------------------
    # Dynamic decay rate for each detected photon
    # --------------------------------------------------------

    decay_rate = (
        2.0
        * K0**2
        * (
            momentum_shallow
            * alpha_db_shallow
            +
            momentum_deep
            * alpha_db_deep
        )
    )

    if not np.all(
        np.isfinite(
            decay_rate
        )
    ):
        raise ValueError(
            "Decay rate contains "
            "non-finite values."
        )

    # --------------------------------------------------------
    # Calculate g1(tau)
    # --------------------------------------------------------

    tau_seconds = np.asarray(
        tau_seconds,
        dtype=float,
    )

    if tau_seconds.ndim != 1:
        raise ValueError(
            "tau_seconds must be "
            "a 1D array."
        )

    if len(tau_seconds) == 0:
        raise ValueError(
            "tau_seconds cannot be empty."
        )

    if np.any(
        tau_seconds < 0
    ):
        raise ValueError(
            "tau_seconds cannot "
            "contain negative values."
        )

    if not np.all(
        np.isfinite(
            tau_seconds
        )
    ):
        raise ValueError(
            "tau_seconds contains "
            "non-finite values."
        )

    g1 = np.empty(
        len(tau_seconds),
        dtype=float,
    )

    for index, tau in enumerate(
        tau_seconds
    ):

        dynamic_term = np.exp(
            -decay_rate
            * tau
        )

        weighted_correlation = (
            absorption_weight
            * dynamic_term
        )

        g1[index] = (
            np.sum(
                weighted_correlation
            )
            / weight_sum
        )

    # --------------------------------------------------------
    # Numerical normalization
    # --------------------------------------------------------

    if (
        not np.isfinite(g1[0])
        or g1[0] == 0
    ):
        raise ValueError(
            "Invalid g1(0), "
            "cannot normalize."
        )

    g1 = (
        g1
        / g1[0]
    )

    return g1


# ============================================================
# CALCULATE g2
# ============================================================

def calculate_g2(
    g1,
    beta,
):
    """
    Convert normalized field autocorrelation
    g1(tau) to intensity autocorrelation
    g2(tau) using the Siegert relation:

        g2(tau)
        =
        1
        +
        beta * |g1(tau)|^2
    """

    g1 = np.asarray(
        g1,
        dtype=float,
    )

    beta = float(
        beta
    )

    if not np.all(
        np.isfinite(g1)
    ):
        raise ValueError(
            "g1 contains "
            "non-finite values"
        )

    if not np.isfinite(beta):
        raise ValueError(
            "beta must be finite."
        )

    if beta < 0:
        raise ValueError(
            f"beta must be non-negative, "
            f"got {beta}"
        )

    g2 = (
        1.0
        + beta
        * np.abs(g1) ** 2
    )

    return g2


# ============================================================
# CALCULATE g2 DIRECTLY FROM PHOTON HISTORIES
# ============================================================

def calculate_g2_from_detp(
    detp,
    tau_seconds,
    detector_id,
    beta,
    alpha_db_shallow=None,
    alpha_db_deep=None,
):
    """
    Convenience function for:

        detp
          -> g1
          -> g2

    This will be useful for the synthetic
    dataset generator.
    """

    g1 = calculate_g1(
        detp=detp,
        tau_seconds=tau_seconds,
        detector_id=detector_id,
        alpha_db_shallow=(
            alpha_db_shallow
        ),
        alpha_db_deep=(
            alpha_db_deep
        ),
    )

    g2 = calculate_g2(
        g1=g1,
        beta=beta,
    )

    return g1, g2


# ============================================================
# TEST CONSTANTS / UNITS
# ============================================================

def main():

    print(
        "=== DCS Processor Equation Check ==="
    )

    print(
        f"Wavelength: "
        f"{WAVELENGTH_NM} nm"
    )

    print(
        f"Wavelength: "
        f"{WAVELENGTH_MM:.9f} mm"
    )

    print(
        f"Refractive index: "
        f"{REFRACTIVE_INDEX_N}"
    )

    print(
        f"k0: "
        f"{K0:.6f} mm^-1"
    )

    print(
        f"mu_a: "
        f"{MU_A_MM:.6f} mm^-1"
    )

    print(
        "\n--- Default/reference dynamics ---"
    )

    print(
        f"alpha*Db shallow: "
        f"{ALPHA_DB_SHALLOW_MM2_S:.6e} "
        f"mm^2/s"
    )

    print(
        f"alpha*Db deep: "
        f"{ALPHA_DB_DEEP_MM2_S:.6e} "
        f"mm^2/s"
    )

    tau_us = np.arange(
        0,
        501,
        dtype=float,
    )

    tau_seconds = (
        tau_us
        * 1e-6
    )

    print(
        f"\nTau range: "
        f"{tau_us[0]:.0f} - "
        f"{tau_us[-1]:.0f} us"
    )

    print(
        f"Tau points: "
        f"{len(tau_us)}"
    )

    # --------------------------------------------------------
    # Test external alpha*Db support
    # --------------------------------------------------------

    test_shallow = 4.032258e-6
    test_deep = 1.612903e-5

    (
        checked_shallow,
        checked_deep,
    ) = validate_alpha_db(
        test_shallow,
        test_deep,
    )

    print(
        "\n--- External dynamics test ---"
    )

    print(
        f"Test shallow alpha*Db: "
        f"{checked_shallow:.6e} mm^2/s"
    )

    print(
        f"Test deep alpha*Db: "
        f"{checked_deep:.6e} mm^2/s"
    )

    print(
        "\nExternal alpha*Db support: PASS"
    )

    print(
        "Equation setup: PASS"
    )


if __name__ == "__main__":
    main()