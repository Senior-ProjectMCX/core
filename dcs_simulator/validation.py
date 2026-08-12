"""
DCS Reference Phantom Validation
================================

Compare synthetic g2 curves generated from the Reference Phantom
simulation against the Physical Phantom reference dataset.

Reference MAT:
    reference_data/Dataset_phantom-intra40*.mat

Expected reference structure:
    Dataset_set_intra40 : (7, 1540)
    ave_gtau_intra40    : (10, 501, 1540)

Reference g2 mapping:
    ave_gtau_intra40[0] -> SDS 1 cm
    ave_gtau_intra40[1] -> SDS 2 cm
    ave_gtau_intra40[2] -> SDS 3 cm

Metrics:
    - MAE
    - RMSE
    - NRMSE (%)
    - Pearson correlation
    - Maximum absolute error

IMPORTANT:
The reference MAT does not contain an explicit tau vector.
This V1 validator therefore compares the 501 g2 samples
by sample index.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_DATA_DIR = BASE_DIR / "reference_data"

OUTPUT_DATASET_DIR = (
    BASE_DIR
    / "output"
    / "datasets"
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

TARGET_SHALLOW_FLOW = 5.0
TARGET_DEEP_FLOW = 20.0

FLOW_TOLERANCE = 1e-8


# Mapping inside ave_gtau_intra40

REFERENCE_G2_INDEX_BY_SDS = {
    1: 0,
    2: 1,
    3: 2,
}


# ============================================================
# FIND FILES
# ============================================================

def find_reference_mat():
    """
    Automatically find Dataset_phantom-intra40 MAT file
    inside reference_data/.
    """

    if not REFERENCE_DATA_DIR.exists():
        raise FileNotFoundError(
            "Reference data directory not found:\n"
            f"{REFERENCE_DATA_DIR}"
        )

    candidates = sorted(
        REFERENCE_DATA_DIR.glob(
            "Dataset_phantom-intra40*.mat"
        )
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            "Cannot find reference MAT file in:\n"
            f"{REFERENCE_DATA_DIR}\n\n"
            "Expected something like:\n"
            "Dataset_phantom-intra40.mat"
        )

    if len(candidates) > 1:
        print(
            "\nWARNING: Multiple reference MAT files found."
        )

        print(
            "Using:"
        )

        print(
            candidates[0]
        )

    return candidates[0]


def find_simulation_csv():
    """
    Find Reference Phantom simulation CSV.

    Human simulation datasets are intentionally ignored.
    """

    if not OUTPUT_DATASET_DIR.exists():
        raise FileNotFoundError(
            "Dataset output directory not found:\n"
            f"{OUTPUT_DATASET_DIR}"
        )

    csv_files = sorted(
        OUTPUT_DATASET_DIR.glob(
            "*.csv"
        )
    )

    # Ignore Human Simulation datasets.

    candidates = [
        path
        for path in csv_files
        if "human" not in path.name.lower()
    ]

    # Prefer files whose names clearly indicate reference/phantom.

    preferred = [
        path
        for path in candidates
        if (
            "reference" in path.name.lower()
            or "phantom" in path.name.lower()
        )
    ]

    if preferred:
        return preferred[0]

    if candidates:
        print(
            "\nWARNING:"
        )

        print(
            "No CSV explicitly named reference/phantom "
            "was found."
        )

        print(
            "Using the first non-human CSV:"
        )

        print(
            candidates[0]
        )

        return candidates[0]

    human_files = [
        path
        for path in csv_files
        if "human" in path.name.lower()
    ]

    message = (
        "\nNo Reference Phantom simulation CSV found.\n\n"
        "Validator needs:\n\n"
        "Physical Phantom MAT\n"
        "        vs.\n"
        "Reference Phantom Simulation CSV\n\n"
        "Expected location:\n"
        f"{OUTPUT_DATASET_DIR}\n\n"
        "Example filename:\n"
        "reference_phantom_dataset.csv\n"
    )

    if human_files:
        message += (
            "\nHuman dataset(s) were found, but they "
            "cannot be used for Reference Phantom "
            "validation:\n"
        )

        for path in human_files:
            message += (
                f"  - {path.name}\n"
            )

    raise FileNotFoundError(
        message
    )


# ============================================================
# LOAD PHYSICAL REFERENCE
# ============================================================

def load_reference_mat(mat_path):
    """
    Load Physical Phantom reference dataset.
    """

    data = loadmat(
        mat_path
    )

    required_keys = [
        "Dataset_set_intra40",
        "ave_gtau_intra40",
    ]

    for key in required_keys:

        if key not in data:
            raise KeyError(
                f"MAT file is missing variable: {key}"
            )

    dataset_set = np.asarray(
        data[
            "Dataset_set_intra40"
        ],
        dtype=float,
    )

    ave_gtau = np.asarray(
        data[
            "ave_gtau_intra40"
        ],
        dtype=float,
    )

    print(
        "\n=== Reference Dataset ==="
    )

    print(
        f"MAT file: {mat_path.name}"
    )

    print(
        "Dataset_set_intra40 shape:",
        dataset_set.shape,
    )

    print(
        "ave_gtau_intra40 shape:",
        ave_gtau.shape,
    )

    if (
        dataset_set.ndim != 2
        or dataset_set.shape[0] < 2
    ):
        raise ValueError(
            "Unexpected Dataset_set_intra40 shape: "
            f"{dataset_set.shape}"
        )

    if (
        ave_gtau.ndim != 3
        or ave_gtau.shape[0] < 3
    ):
        raise ValueError(
            "Unexpected ave_gtau_intra40 shape: "
            f"{ave_gtau.shape}"
        )

    if (
        dataset_set.shape[1]
        != ave_gtau.shape[2]
    ):
        raise ValueError(
            "Reference metadata and g2 record counts "
            "do not match."
        )

    return (
        dataset_set,
        ave_gtau,
    )


# ============================================================
# FIND MATCHING PHYSICAL TRIALS
# ============================================================

def find_reference_columns(
    dataset_set,
    shallow_flow,
    deep_flow,
):
    """
    Find reference records matching the commanded
    shallow/deep flow condition.

    Dataset_set_intra40:
        row 0 -> commanded shallow flow
        row 1 -> commanded deep flow
    """

    shallow_values = dataset_set[
        0,
        :
    ]

    deep_values = dataset_set[
        1,
        :
    ]

    shallow_match = np.isclose(
        shallow_values,
        shallow_flow,
        atol=FLOW_TOLERANCE,
        rtol=0.0,
    )

    deep_match = np.isclose(
        deep_values,
        deep_flow,
        atol=FLOW_TOLERANCE,
        rtol=0.0,
    )

    matching_columns = np.where(
        shallow_match
        & deep_match
    )[0]

    if len(matching_columns) == 0:
        raise ValueError(
            "Physical reference condition not found:\n"
            f"Shallow = {shallow_flow:g}\n"
            f"Deep    = {deep_flow:g}"
        )

    return matching_columns


# ============================================================
# REFERENCE CURVES
# ============================================================

def extract_reference_trial_curves(
    ave_gtau,
    matching_columns,
    g2_index,
):
    """
    Extract reference trial curves and normalize
    the result to:

        number_of_trials x number_of_tau_points
    """

    curves = ave_gtau[
        g2_index,
        :,
        matching_columns,
    ]

    curves = np.asarray(
        curves,
        dtype=float,
    )

    if curves.ndim != 2:
        raise ValueError(
            "Unexpected reference curve shape: "
            f"{curves.shape}"
        )

    tau_points = ave_gtau.shape[1]
    trial_count = len(
        matching_columns
    )

    if curves.shape == (
        trial_count,
        tau_points,
    ):
        return curves

    if curves.shape == (
        tau_points,
        trial_count,
    ):
        return curves.T

    raise ValueError(
        "Cannot determine reference trial/tau "
        "dimensions.\n"
        f"Curve shape: {curves.shape}\n"
        f"Expected trials: {trial_count}\n"
        f"Expected tau points: {tau_points}"
    )


def get_reference_curves(
    ave_gtau,
    matching_columns,
):
    """
    Calculate mean and standard deviation across
    Physical Phantom trials for each SDS.
    """

    reference = {}

    for (
        sds_cm,
        g2_index,
    ) in REFERENCE_G2_INDEX_BY_SDS.items():

        trial_curves = (
            extract_reference_trial_curves(
                ave_gtau=ave_gtau,
                matching_columns=matching_columns,
                g2_index=g2_index,
            )
        )

        mean_curve = np.mean(
            trial_curves,
            axis=0,
        )

        std_curve = np.std(
            trial_curves,
            axis=0,
            ddof=1,
        )

        reference[
            sds_cm
        ] = {
            "trials": trial_curves,
            "mean": mean_curve,
            "std": std_curve,
        }

    return reference


# ============================================================
# LOAD SYNTHETIC DATASET
# ============================================================

def load_simulation_csv(csv_path):
    """
    Load Reference Phantom synthetic dataset.
    """

    dataset = pd.read_csv(
        csv_path
    )

    required_columns = [
        "flow_shallow",
        "flow_deep",
        "sds_cm",
        "g2",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            "Simulation CSV is missing columns:\n"
            + ", ".join(
                missing
            )
        )

    print(
        "\n=== Simulation Dataset ==="
    )

    print(
        f"CSV file: {csv_path.name}"
    )

    print(
        f"Rows: {len(dataset):,}"
    )

    return dataset


# ============================================================
# GET ONE SYNTHETIC CURVE
# ============================================================

def get_simulation_curve(
    dataset,
    shallow_flow,
    deep_flow,
    sds_cm,
):
    """
    Extract one simulated g2 curve.
    """

    shallow_match = np.isclose(
        dataset[
            "flow_shallow"
        ].to_numpy(
            dtype=float
        ),
        shallow_flow,
        atol=FLOW_TOLERANCE,
        rtol=0.0,
    )

    deep_match = np.isclose(
        dataset[
            "flow_deep"
        ].to_numpy(
            dtype=float
        ),
        deep_flow,
        atol=FLOW_TOLERANCE,
        rtol=0.0,
    )

    sds_match = np.isclose(
        dataset[
            "sds_cm"
        ].to_numpy(
            dtype=float
        ),
        sds_cm,
        atol=FLOW_TOLERANCE,
        rtol=0.0,
    )

    selected = dataset.loc[
        shallow_match
        & deep_match
        & sds_match
    ].copy()

    if selected.empty:
        raise ValueError(
            "Simulation condition not found:\n"
            f"Shallow = {shallow_flow:g}\n"
            f"Deep    = {deep_flow:g}\n"
            f"SDS     = {sds_cm} cm"
        )

    if "tau_us" in selected.columns:

        selected = selected.sort_values(
            "tau_us"
        )

    curve = selected[
        "g2"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            curve
        )
    ):
        raise ValueError(
            "Simulation g2 curve contains "
            "NaN or infinite values."
        )

    return curve


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    reference,
    simulation,
):
    """
    Compare synthetic g2 against the mean
    Physical Phantom g2 curve.

    NRMSE is normalized by reference dynamic range:

        RMSE
        --------------------- x 100
        max(ref) - min(ref)

    NRMSE is an ERROR percentage.
    It is NOT an accuracy percentage.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    simulation = np.asarray(
        simulation,
        dtype=float,
    )

    if reference.shape != simulation.shape:
        raise ValueError(
            "Reference and simulation curves "
            "have different shapes:\n"
            f"Reference:  {reference.shape}\n"
            f"Simulation: {simulation.shape}"
        )

    if not np.all(
        np.isfinite(
            reference
        )
    ):
        raise ValueError(
            "Reference curve contains "
            "NaN or infinite values."
        )

    error = (
        simulation
        - reference
    )

    absolute_error = np.abs(
        error
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    reference_range = float(
        np.max(
            reference
        )
        - np.min(
            reference
        )
    )

    if reference_range > 0.0:

        nrmse_percent = (
            rmse
            / reference_range
            * 100.0
        )

    else:

        nrmse_percent = np.nan

    reference_std = float(
        np.std(
            reference
        )
    )

    simulation_std = float(
        np.std(
            simulation
        )
    )

    if (
        reference_std > 0.0
        and simulation_std > 0.0
    ):

        correlation = float(
            np.corrcoef(
                reference,
                simulation,
            )[0, 1]
        )

    else:

        correlation = np.nan

    max_absolute_error = float(
        np.max(
            absolute_error
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse_percent": (
            nrmse_percent
        ),
        "correlation": (
            correlation
        ),
        "max_absolute_error": (
            max_absolute_error
        ),
    }


# ============================================================
# VALIDATE ONE CONDITION
# ============================================================

def validate_condition(
    reference_mat,
    simulation_csv,
    shallow_flow,
    deep_flow,
):
    """
    Validate SDS 1, 2 and 3 cm for one
    shallow/deep flow condition.
    """

    (
        dataset_set,
        ave_gtau,
    ) = load_reference_mat(
        reference_mat
    )

    matching_columns = (
        find_reference_columns(
            dataset_set=dataset_set,
            shallow_flow=shallow_flow,
            deep_flow=deep_flow,
        )
    )

    reference_curves = (
        get_reference_curves(
            ave_gtau=ave_gtau,
            matching_columns=matching_columns,
        )
    )

    simulation_dataset = (
        load_simulation_csv(
            simulation_csv
        )
    )

    print(
        "\n======================================"
    )

    print(
        "DCS REFERENCE PHANTOM VALIDATION"
    )

    print(
        "======================================"
    )

    print(
        f"\nShallow flow : "
        f"{shallow_flow:g}"
    )

    print(
        f"Deep flow    : "
        f"{deep_flow:g}"
    )

    print(
        f"Reference trials found: "
        f"{len(matching_columns)}"
    )

    print(
        "Reference columns: "
        + ", ".join(
            str(
                int(index)
            )
            for index
            in matching_columns
        )
    )

    print(
        "\nNOTE:"
    )

    print(
        "Reference MAT has no explicit tau vector."
    )

    print(
        "V1 compares the 501 g2 samples "
        "by sample index."
    )

    results = []

    for sds_cm in [
        1,
        2,
        3,
    ]:

        reference_curve = (
            reference_curves[
                sds_cm
            ][
                "mean"
            ]
        )

        simulation_curve = (
            get_simulation_curve(
                dataset=simulation_dataset,
                shallow_flow=shallow_flow,
                deep_flow=deep_flow,
                sds_cm=sds_cm,
            )
        )

        if (
            len(reference_curve)
            != len(simulation_curve)
        ):
            raise ValueError(
                f"\nSDS {sds_cm} cm has "
                "different sample counts:\n"
                f"Reference  = "
                f"{len(reference_curve)}\n"
                f"Simulation = "
                f"{len(simulation_curve)}\n\n"
                "Do NOT interpolate automatically yet. "
                "The reference tau definition must "
                "be confirmed first."
            )

        metrics = calculate_metrics(
            reference=reference_curve,
            simulation=simulation_curve,
        )

        results.append(
            {
                "sds_cm": sds_cm,
                **metrics,
            }
        )

        print(
            f"\n--- SDS {sds_cm} cm ---"
        )

        print(
            f"Points:       "
            f"{len(reference_curve)}"
        )

        print(
            f"Reference g2: "
            f"{reference_curve[0]:.6f}"
            f" -> "
            f"{reference_curve[-1]:.6f}"
        )

        print(
            f"Simulation g2:"
            f" {simulation_curve[0]:.6f}"
            f" -> "
            f"{simulation_curve[-1]:.6f}"
        )

        print(
            f"MAE:          "
            f"{metrics['mae']:.8f}"
        )

        print(
            f"RMSE:         "
            f"{metrics['rmse']:.8f}"
        )

        print(
            f"NRMSE:        "
            f"{metrics['nrmse_percent']:.3f} %"
        )

        print(
            f"Correlation:  "
            f"{metrics['correlation']:.6f}"
        )

        print(
            f"Max Abs Err:  "
            f"{metrics['max_absolute_error']:.8f}"
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n======================================"
    )

    print(
        "OVERALL SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"Mean MAE:         "
        f"{results_df['mae'].mean():.8f}"
    )

    print(
        f"Mean RMSE:        "
        f"{results_df['rmse'].mean():.8f}"
    )

    print(
        f"Mean NRMSE:       "
        f"{results_df['nrmse_percent'].mean():.3f} %"
    )

    print(
        f"Mean Correlation: "
        f"{results_df['correlation'].mean():.6f}"
    )

    worst_index = results_df[
        "nrmse_percent"
    ].idxmax()

    worst_row = results_df.loc[
        worst_index
    ]

    print(
        f"\nWorst SDS:        "
        f"{int(worst_row['sds_cm'])} cm"
    )

    print(
        f"Worst NRMSE:      "
        f"{worst_row['nrmse_percent']:.3f} %"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== DCS Validation Tool ==="
    )

    reference_mat = (
        find_reference_mat()
    )

    print(
        "\nReference MAT:"
    )

    print(
        reference_mat
    )

    simulation_csv = (
        find_simulation_csv()
    )

    print(
        "\nReference Simulation CSV:"
    )

    print(
        simulation_csv
    )

    results = validate_condition(
        reference_mat=reference_mat,
        simulation_csv=simulation_csv,
        shallow_flow=TARGET_SHALLOW_FLOW,
        deep_flow=TARGET_DEEP_FLOW,
    )

    print(
        "\n======================================"
    )

    print(
        "Validation completed."
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()