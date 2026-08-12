"""
Reference Phantom Synthetic DCS Dataset Generator.

Pipeline
--------
Reference Phantom Geometry
    -> MCX photon simulation
    -> full detected-photon histories (DPM)
    -> shallow/deep flow condition
    -> alpha*Db
    -> g1
    -> g2
    -> reference_phantom_dataset.csv

IMPORTANT
---------
This generator uses the Reference Phantom dynamics
defined in parameters.py.

It does NOT use the Human muscle gamma calibration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pmcx

from parameters import (
    SHALLOW_FLOW_ML_MIN_100G,
    DEEP_FLOW_ML_MIN_100G,
    FLOW_TO_ALPHA_DB,
    TAU_US,
    TAU_S,
)

from simulation import (
    create_test_config,
)

from dcs_processor import (
    calculate_g2_from_detp,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
)

DATASET_DIR = (
    OUTPUT_DIR
    / "datasets"
)

DETP_FILE = (
    OUTPUT_DIR
    / "detp_core_v1.npy"
)

OUTPUT_CSV = (
    DATASET_DIR
    / "reference_phantom_dataset.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# beta values currently used by the project
# for SDS 1, 2 and 3 cm.

BETA_BY_DETECTOR = {
    1: 1.43009039 - 1.0,
    2: 1.50906565 - 1.0,
    3: 1.45281563 - 1.0,
}

SDS_CM_BY_DETECTOR = {
    1: 1,
    2: 2,
    3: 3,
}


# ============================================================
# REFERENCE FLOW -> ALPHA*DB
# ============================================================

def reference_flow_to_alpha_db(
    flow,
):
    """
    Reference Phantom Core V1 conversion:

        alpha*Db
        =
        flow * FLOW_TO_ALPHA_DB
    """

    flow = float(
        flow
    )

    if flow < 0:
        raise ValueError(
            "Flow cannot be negative."
        )

    return (
        flow
        * FLOW_TO_ALPHA_DB
    )


# ============================================================
# VALIDATE DETP
# ============================================================

def validate_detp(
    detp,
):
    """
    Validate PMCX DPM output.

    Expected:
        Row 0 = detector ID
        Row 1 = shallow pathlength
        Row 2 = deep pathlength
        Row 3 = shallow momentum
        Row 4 = deep momentum
    """

    detp = np.asarray(
        detp
    )

    if detp.ndim != 2:
        raise ValueError(
            f"detp must be 2D, got "
            f"{detp.shape}"
        )

    if detp.shape[0] != 5:
        raise ValueError(
            "Expected DPM detp with 5 rows, "
            f"got {detp.shape[0]}"
        )

    if detp.shape[1] == 0:
        raise ValueError(
            "detp contains no detected photons."
        )

    if not np.all(
        np.isfinite(
            detp
        )
    ):
        raise ValueError(
            "detp contains non-finite values."
        )

    return detp


# ============================================================
# LOAD OR GENERATE REFERENCE PHOTONS
# ============================================================

def load_or_generate_detp():
    """
    Reuse the saved full Reference Phantom photon
    histories when available.

    If no full DPM baseline exists, run MCX once
    and save it for future tuning.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reuse existing full baseline
    # --------------------------------------------------------

    if DETP_FILE.exists():

        print(
            "=== Loading Existing Reference DPM Baseline ==="
        )

        print(
            DETP_FILE
        )

        detp = np.load(
            DETP_FILE
        )

        validate_detp(
            detp
        )

        print(
            f"Detected photons: "
            f"{detp.shape[1]:,}"
        )

        return detp

    # --------------------------------------------------------
    # Otherwise run MCX once
    # --------------------------------------------------------

    print(
        "=== Reference DPM Baseline Not Found ==="
    )

    print(
        "Running Reference Phantom MCX once..."
    )

    cfg = (
        create_test_config()
    )

    print(
        f"Photons launched: "
        f"{cfg['nphoton']:,}"
    )

    result = pmcx.run(
        cfg
    )

    if "detp" not in result:
        raise ValueError(
            "PMCX did not return detp."
        )

    detp = validate_detp(
        result[
            "detp"
        ]
    )

    np.save(
        DETP_FILE,
        detp,
    )

    print(
        "\nReference MCX: PASS"
    )

    print(
        f"Detected photons: "
        f"{detp.shape[1]:,}"
    )

    print(
        f"Full DPM baseline saved to:\n"
        f"{DETP_FILE}"
    )

    return detp


# ============================================================
# PRINT DETECTOR SUMMARY
# ============================================================

def print_detector_summary(
    detp,
):
    """
    Show detected photon counts for all SDS.
    """

    detector_ids = (
        detp[0]
        .astype(int)
    )

    print(
        "\n=== Detected Photon Summary ==="
    )

    for detector_id in (
        1,
        2,
        3,
    ):

        count = int(
            np.sum(
                detector_ids
                == detector_id
            )
        )

        print(
            f"SDS "
            f"{SDS_CM_BY_DETECTOR[detector_id]} cm: "
            f"{count:,} photons"
        )


# ============================================================
# GENERATE REFERENCE DATASET
# ============================================================

def generate_reference_dataset(
    detp,
    shallow_flow,
    deep_flow,
):
    """
    Generate synthetic g2 curves for the
    Reference Phantom condition.
    """

    alpha_db_shallow = (
        reference_flow_to_alpha_db(
            shallow_flow
        )
    )

    alpha_db_deep = (
        reference_flow_to_alpha_db(
            deep_flow
        )
    )

    print(
        "\n=== Reference Flow Condition ==="
    )

    print(
        f"Shallow flow: "
        f"{shallow_flow:g}"
    )

    print(
        f"Deep flow:    "
        f"{deep_flow:g}"
    )

    print(
        "\n=== Reference Dynamics ==="
    )

    print(
        f"FLOW_TO_ALPHA_DB: "
        f"{FLOW_TO_ALPHA_DB:.6e}"
    )

    print(
        f"alpha*Db shallow: "
        f"{alpha_db_shallow:.6e} mm^2/s"
    )

    print(
        f"alpha*Db deep:    "
        f"{alpha_db_deep:.6e} mm^2/s"
    )

    rows = []

    for detector_id in (
        1,
        2,
        3,
    ):

        sds_cm = (
            SDS_CM_BY_DETECTOR[
                detector_id
            ]
        )

        beta = (
            BETA_BY_DETECTOR[
                detector_id
            ]
        )

        print(
            f"\n--- SDS {sds_cm} cm ---"
        )

        print(
            f"beta: {beta:.8f}"
        )

        (
            g1,
            g2,
        ) = calculate_g2_from_detp(
            detp=detp,
            tau_seconds=TAU_S,
            detector_id=detector_id,
            beta=beta,
            alpha_db_shallow=(
                alpha_db_shallow
            ),
            alpha_db_deep=(
                alpha_db_deep
            ),
        )

        if len(g2) != len(
            TAU_US
        ):
            raise ValueError(
                f"SDS {sds_cm} cm returned "
                f"{len(g2)} points, expected "
                f"{len(TAU_US)}."
            )

        if not np.all(
            np.isfinite(
                g2
            )
        ):
            raise ValueError(
                f"SDS {sds_cm} cm "
                "contains non-finite g2."
            )

        print(
            f"g1: "
            f"{g1[0]:.6f} "
            f"-> {g1[-1]:.6f}"
        )

        print(
            f"g2: "
            f"{g2[0]:.6f} "
            f"-> {g2[-1]:.6f}"
        )

        for (
            tau_us,
            g2_value,
        ) in zip(
            TAU_US,
            g2,
        ):

            rows.append(
                {
                    "flow_shallow":
                        float(
                            shallow_flow
                        ),

                    "flow_deep":
                        float(
                            deep_flow
                        ),

                    "sds_cm":
                        int(
                            sds_cm
                        ),

                    "tau_us":
                        float(
                            tau_us
                        ),

                    "g2":
                        float(
                            g2_value
                        ),
                }
            )

    dataset = (
        pd.DataFrame(
            rows
        )
    )

    return dataset


# ============================================================
# VALIDATE GENERATED DATASET
# ============================================================

def validate_generated_dataset(
    dataset,
):
    """
    Sanity-check exported Reference Phantom dataset.
    """

    expected_rows = (
        3
        * len(
            TAU_US
        )
    )

    if len(
        dataset
    ) != expected_rows:

        raise ValueError(
            f"Generated {len(dataset)} rows, "
            f"expected {expected_rows}."
        )

    required_columns = [
        "flow_shallow",
        "flow_deep",
        "sds_cm",
        "tau_us",
        "g2",
    ]

    for column in required_columns:

        if column not in dataset.columns:
            raise ValueError(
                f"Missing column: "
                f"{column}"
            )

    if not np.all(
        np.isfinite(
            dataset[
                "g2"
            ]
        )
    ):
        raise ValueError(
            "Generated dataset contains "
            "non-finite g2 values."
        )

    if np.any(
        dataset[
            "g2"
        ]
        < 1.0
    ):
        raise ValueError(
            "Generated dataset contains "
            "g2 below 1."
        )

    print(
        "\nDataset validation: PASS"
    )

    print(
        f"Rows generated: "
        f"{len(dataset):,}"
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    dataset,
):
    """
    Export Reference Phantom synthetic dataset.
    """

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "REFERENCE PHANTOM DATASET GENERATOR"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Full reference photon histories
    # --------------------------------------------------------

    detp = (
        load_or_generate_detp()
    )

    print_detector_summary(
        detp
    )

    # --------------------------------------------------------
    # Generate g2 for reference condition
    # --------------------------------------------------------

    dataset = (
        generate_reference_dataset(
            detp=detp,
            shallow_flow=(
                SHALLOW_FLOW_ML_MIN_100G
            ),
            deep_flow=(
                DEEP_FLOW_ML_MIN_100G
            ),
        )
    )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    validate_generated_dataset(
        dataset
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    save_dataset(
        dataset
    )

    print(
        "\n======================================"
    )

    print(
        "Reference Phantom Dataset: PASS"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()