"""
Flow-rate conversion for Human DCS Simulator V1.

Human V1 uses a population-averaged skeletal-muscle
calibration coefficient to convert absolute flow rate
to DCS blood-flow index (BFI / alpha*Db).

IMPORTANT
---------
This conversion is used only for the Human Synthetic
Dataset Generator.

It is NOT the calibration model for the physical phantom.

Relationship:

    BF = gamma * BFI

therefore:

    BFI = BF / gamma

Reference calibration coefficient:

    gamma = 1.24e8

BF unit:
    mL / 100 mL / min

BFI unit:
    cm^2 / s

The DCS processing code uses mm^2 / s, therefore:

    1 cm^2 = 100 mm^2
"""


# ============================================================
# CALIBRATION CONSTANT
# ============================================================

HUMAN_MUSCLE_GAMMA = 1.24e8


# ============================================================
# FLOW -> BFI
# ============================================================

def flow_to_bfi_cm2_s(
    flow_ml_100ml_min,
    gamma=HUMAN_MUSCLE_GAMMA,
):
    """
    Convert skeletal-muscle flow rate to DCS BFI.

    Parameters
    ----------
    flow_ml_100ml_min : float
        Flow in mL / 100 mL / min.

    gamma : float
        Calibration coefficient.

    Returns
    -------
    float
        BFI in cm^2 / s.
    """

    flow = float(
        flow_ml_100ml_min
    )

    if flow < 0:
        raise ValueError(
            "Flow rate cannot be negative."
        )

    if gamma <= 0:
        raise ValueError(
            "Calibration coefficient "
            "must be greater than zero."
        )

    return flow / gamma


# ============================================================
# BFI -> MM^2/S
# ============================================================

def bfi_cm2_s_to_mm2_s(
    bfi_cm2_s,
):
    """
    Convert BFI from cm^2/s to mm^2/s.
    """

    bfi = float(
        bfi_cm2_s
    )

    return bfi * 100.0


# ============================================================
# FLOW -> ALPHA*DB
# ============================================================

def flow_to_alpha_db_mm2_s(
    flow_ml_100ml_min,
    gamma=HUMAN_MUSCLE_GAMMA,
):
    """
    Convert flow rate directly to the alpha*Db value
    used by the DCS processor.

    Returns
    -------
    float
        alpha*Db in mm^2/s.
    """

    bfi_cm2_s = (
        flow_to_bfi_cm2_s(
            flow_ml_100ml_min,
            gamma=gamma,
        )
    )

    return (
        bfi_cm2_s_to_mm2_s(
            bfi_cm2_s
        )
    )


# ============================================================
# TEST
# ============================================================

def main():

    print(
        "=== Human Flow Conversion Test ==="
    )

    print(
        f"Calibration gamma: "
        f"{HUMAN_MUSCLE_GAMMA:.3e}"
    )

    test_flows = [
        5.0,
        10.0,
        15.0,
        20.0,
    ]

    print(
        "\nFlow -> BFI -> alpha*Db"
    )

    for flow in test_flows:

        bfi = flow_to_bfi_cm2_s(
            flow
        )

        alpha_db = (
            flow_to_alpha_db_mm2_s(
                flow
            )
        )

        print(
            f"\nFlow: "
            f"{flow:.1f} "
            f"mL/100mL/min"
        )

        print(
            f"BFI:      "
            f"{bfi:.6e} cm^2/s"
        )

        print(
            f"alpha*Db: "
            f"{alpha_db:.6e} mm^2/s"
        )

    # --------------------------------------------------------
    # Known-value sanity check
    # --------------------------------------------------------

    expected_for_5 = (
        5.0
        / HUMAN_MUSCLE_GAMMA
        * 100.0
    )

    actual_for_5 = (
        flow_to_alpha_db_mm2_s(
            5.0
        )
    )

    if abs(
        actual_for_5
        - expected_for_5
    ) > 1e-15:

        raise ValueError(
            "Flow conversion sanity "
            "check failed."
        )

    print(
        "\nHuman flow conversion: PASS"
    )


if __name__ == "__main__":
    main()