"""
Synthetic DCS dataset generator for Human Thigh Model V1.

Pipeline:
Human input
    -> subject-specific superficial geometry
    -> one MCX photon simulation
    -> reuse detected-photon histories
    -> multiple shallow/deep flow conditions
    -> flow -> alpha*Db
    -> g1 -> g2
    -> tabular synthetic dataset
    -> CSV export

Human Model V1
--------------
Required:
- Sex
- Measured thigh SAT thickness

Optional:
- Measured skin thickness

If measured skin thickness is not supplied,
a sex-based literature reference is used.

IMPORTANT
---------
- MCX is run once per human geometry.
- Different flow conditions reuse the same detected-photon
  histories and momentum-transfer information.
- SAT is treated as a measured subject-specific parameter.
- Sex is used only to select the fallback skin-thickness
  reference when measured skin thickness is unavailable.
- Human V1 is a simulation prototype, not a validated
  individualized blood-flow estimator.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pmcx

from human_model import (
    HumanInput,
    build_human_thigh_geometry,
)

from phantom import (
    create_two_layer_phantom,
    create_mcx_properties,
    validate_phantom,
    get_layer_info,
)

from flow_conversion import (
    flow_to_alpha_db_mm2_s,
)

from dcs_processor import (
    calculate_g2_from_detp,
)

from parameters import (
    NX,
    NY,
    VOXEL_SIZE_MM,
)


# ============================================================
# SETTINGS
# ============================================================

NPHOTON_DATASET = 10_000_000

SOURCE_POS = [
    NX / 2,
    NY / 2,
    0.0,
]

SOURCE_DIR = [
    0.0,
    0.0,
    1.0,
]

DETECTOR_RADIUS_VOXELS = 3.0

TAU_US = np.arange(
    0,
    501,
    dtype=float,
)

TAU_SECONDS = (
    TAU_US
    * 1e-6
)

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
# DETECTORS
# ============================================================

def build_detector_positions():

    source_x = SOURCE_POS[0]
    source_y = SOURCE_POS[1]

    detector_z = -0.5

    return [
        [
            source_x + 20.0,
            source_y,
            detector_z,
            DETECTOR_RADIUS_VOXELS,
        ],
        [
            source_x + 40.0,
            source_y,
            detector_z,
            DETECTOR_RADIUS_VOXELS,
        ],
        [
            source_x + 60.0,
            source_y,
            detector_z,
            DETECTOR_RADIUS_VOXELS,
        ],
    ]


# ============================================================
# MCX CONFIG
# ============================================================

def create_human_mcx_config(
    volume,
    nphoton=NPHOTON_DATASET,
):

    validate_phantom(
        volume
    )

    cfg = {
        "nphoton": int(nphoton),

        "vol": volume,

        "prop": (
            create_mcx_properties()
        ),

        "srcpos": SOURCE_POS,
        "srcdir": SOURCE_DIR,

        "detpos": (
            build_detector_positions()
        ),

        "issavedet": 1,

        # D = detector ID
        # P = partial pathlength
        # M = momentum transfer
        "savedetflag": "DPM",

        "unitinmm": VOXEL_SIZE_MM,

        "tstart": 0.0,
        "tend": 5e-9,
        "tstep": 5e-9,

        "autopilot": 1,
        "gpuid": 1,
    }

    return cfg


# ============================================================
# RUN ONE HUMAN MCX SIMULATION
# ============================================================

def simulate_human_photons(
    human,
    nphoton=NPHOTON_DATASET,
):
    """
    Create subject-specific geometry and run MCX once.

    Returns:
        geometry
        layer_info
        detected-photon histories
    """

    geometry = (
        build_human_thigh_geometry(
            human
        )
    )

    volume = (
        create_two_layer_phantom(
            shallow_depth_mm=(
                geometry
                .superficial_thickness_mm
            )
        )
    )

    validate_phantom(
        volume
    )

    layer_info = (
        get_layer_info(
            volume
        )
    )

    cfg = (
        create_human_mcx_config(
            volume=volume,
            nphoton=nphoton,
        )
    )

    print(
        "=== Human MCX Dataset Simulation ==="
    )

    print(
        f"Sex: "
        f"{geometry.sex}"
    )

    print(
        f"Skin thickness: "
        f"{geometry.skin_thickness_mm:.2f} mm"
    )

    print(
        f"Skin source: "
        f"{geometry.skin_thickness_source}"
    )

    print(
        f"SAT thickness: "
        f"{geometry.sat_thickness_mm:.2f} mm"
    )

    print(
        f"Requested superficial depth: "
        f"{geometry.superficial_thickness_mm:.2f} mm"
    )

    print(
        f"Voxelized superficial depth: "
        f"{layer_info['shallow_depth_mm']:.2f} mm"
    )

    voxelization_difference = abs(
        geometry.superficial_thickness_mm
        - layer_info["shallow_depth_mm"]
    )

    print(
        f"Voxelization difference: "
        f"{voxelization_difference:.2f} mm"
    )

    if voxelization_difference > VOXEL_SIZE_MM:
        raise ValueError(
            "Human superficial depth changed "
            "too much during voxelization."
        )

    print(
        f"Photons: "
        f"{cfg['nphoton']}"
    )

    print(
        "\nRunning MCX once for this human geometry..."
    )

    result = pmcx.run(
        cfg
    )

    if "detp" not in result:
        raise ValueError(
            "PMCX did not return detected-photon data."
        )

    detp = result[
        "detp"
    ]

    if (
        detp.ndim != 2
        or detp.shape[0] != 5
    ):
        raise ValueError(
            f"Unexpected detp shape: "
            f"{detp.shape}"
        )

    print(
        "\nMCX photon simulation: PASS"
    )

    print(
        f"Detected photons: "
        f"{detp.shape[1]}"
    )

    return (
        geometry,
        layer_info,
        detp,
    )


# ============================================================
# GENERATE DATASET FROM ONE PHOTON HISTORY SET
# ============================================================

def generate_dataset_from_detp(
    human,
    geometry,
    layer_info,
    detp,
    shallow_flows,
    deep_flows,
):
    """
    Generate g2 curves for every combination of:

        shallow flow x deep flow x SDS x tau

    without rerunning MCX.
    """

    rows = []

    shallow_flows = [
        float(value)
        for value in shallow_flows
    ]

    deep_flows = [
        float(value)
        for value in deep_flows
    ]

    total_conditions = (
        len(shallow_flows)
        * len(deep_flows)
    )

    condition_index = 0

    print(
        "\n=== Generating Flow Conditions ==="
    )

    for shallow_flow in shallow_flows:

        alpha_db_shallow = (
            flow_to_alpha_db_mm2_s(
                shallow_flow
            )
        )

        for deep_flow in deep_flows:

            condition_index += 1

            alpha_db_deep = (
                flow_to_alpha_db_mm2_s(
                    deep_flow
                )
            )

            print(
                f"Condition "
                f"{condition_index}/"
                f"{total_conditions}: "
                f"Shallow={shallow_flow:g}, "
                f"Deep={deep_flow:g}"
            )

            for detector_id in (
                1,
                2,
                3,
            ):

                beta = (
                    BETA_BY_DETECTOR[
                        detector_id
                    ]
                )

                _, g2 = (
                    calculate_g2_from_detp(
                        detp=detp,
                        tau_seconds=(
                            TAU_SECONDS
                        ),
                        detector_id=(
                            detector_id
                        ),
                        beta=beta,
                        alpha_db_shallow=(
                            alpha_db_shallow
                        ),
                        alpha_db_deep=(
                            alpha_db_deep
                        ),
                    )
                )

                sds_cm = (
                    SDS_CM_BY_DETECTOR[
                        detector_id
                    ]
                )

                for tau_index in range(
                    len(TAU_US)
                ):

                    rows.append(
                        {
                            "sex":
                                geometry.sex,

                            "skin_thickness_mm":
                                geometry
                                .skin_thickness_mm,

                            "skin_thickness_source":
                                geometry
                                .skin_thickness_source,

                            "sat_thickness_mm":
                                geometry
                                .sat_thickness_mm,

                            "requested_superficial_depth_mm":
                                geometry
                                .superficial_thickness_mm,

                            "voxelized_superficial_depth_mm":
                                layer_info[
                                    "shallow_depth_mm"
                                ],

                            "deep_region_depth_mm":
                                layer_info[
                                    "deep_depth_mm"
                                ],

                            "flow_shallow":
                                shallow_flow,

                            "flow_deep":
                                deep_flow,

                            "alpha_db_shallow_mm2_s":
                                alpha_db_shallow,

                            "alpha_db_deep_mm2_s":
                                alpha_db_deep,

                            "sds_cm":
                                sds_cm,

                            "tau_us":
                                TAU_US[
                                    tau_index
                                ],

                            "g2":
                                g2[
                                    tau_index
                                ],
                        }
                    )

    dataset = (
        pd.DataFrame(
            rows
        )
    )

    return dataset


# ============================================================
# EXPORT
# ============================================================

def export_dataset(
    dataset,
    filename=None,
):
    """
    Export generated dataset to CSV.
    """

    output_dir = (
        Path(__file__)
        .resolve()
        .parent
        / "output"
        / "datasets"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if filename is None:
        filename = (
            "human_dcs_dataset_v1.csv"
        )

    output_file = (
        output_dir
        / filename
    )

    dataset.to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================
# DATASET SANITY CHECK
# ============================================================

def validate_dataset(
    dataset,
    shallow_flows,
    deep_flows,
):
    """
    Check expected dataset size and numerical validity.
    """

    expected_rows = (
        len(shallow_flows)
        * len(deep_flows)
        * 3
        * len(TAU_US)
    )

    if len(dataset) != expected_rows:
        raise ValueError(
            f"Dataset row count is "
            f"{len(dataset)}, "
            f"expected {expected_rows}"
        )

    if not np.all(
        np.isfinite(
            dataset["g2"]
        )
    ):
        raise ValueError(
            "Dataset contains non-finite g2 values."
        )

    if np.any(
        dataset["g2"] < 1.0
    ):
        raise ValueError(
            "Dataset contains g2 values below 1."
        )

    required_columns = [
        "sex",
        "skin_thickness_mm",
        "skin_thickness_source",
        "sat_thickness_mm",
        "requested_superficial_depth_mm",
        "voxelized_superficial_depth_mm",
        "flow_shallow",
        "flow_deep",
        "sds_cm",
        "tau_us",
        "g2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    return True


# ============================================================
# FULL HUMAN DATASET GENERATOR
# ============================================================

def generate_human_dataset(
    human,
    shallow_flows,
    deep_flows,
    nphoton=NPHOTON_DATASET,
    export=True,
    filename=None,
):
    """
    Complete pipeline:

        Human
        -> MCX once
        -> multiple flow conditions
        -> synthetic g2 dataset
        -> optional CSV
    """

    (
        geometry,
        layer_info,
        detp,
    ) = simulate_human_photons(
        human=human,
        nphoton=nphoton,
    )

    dataset = (
        generate_dataset_from_detp(
            human=human,
            geometry=geometry,
            layer_info=layer_info,
            detp=detp,
            shallow_flows=(
                shallow_flows
            ),
            deep_flows=(
                deep_flows
            ),
        )
    )

    validate_dataset(
        dataset=dataset,
        shallow_flows=(
            shallow_flows
        ),
        deep_flows=(
            deep_flows
        ),
    )

    print(
        "\nDataset validation: PASS"
    )

    print(
        f"Rows generated: "
        f"{len(dataset)}"
    )

    print(
        "\nDataset preview:"
    )

    print(
        dataset.head(10)
        .to_string(index=False)
    )

    output_file = None

    if export:

        output_file = (
            export_dataset(
                dataset=dataset,
                filename=filename,
            )
        )

        print(
            f"\nDataset saved to:"
            f"\n{output_file}"
        )

    print(
        "\n======================================"
    )

    print(
        "Human -> MCX -> multi-flow g2 "
        "dataset: PASS"
    )

    print(
        "======================================"
    )

    return {
        "dataset": dataset,
        "geometry": geometry,
        "layer_info": layer_info,
        "detp": detp,
        "output_file": output_file,
    }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    # --------------------------------------------------------
    # Test subject
    #
    # Skin thickness is intentionally omitted here.
    # Female -> sex-based reference skin thickness.
    # --------------------------------------------------------

    human = HumanInput(
        sex="Female",
        sat_thickness_mm=10.0,
    )

    # Small V1 test grid:
    #
    # 3 shallow values
    # x
    # 3 deep values
    #
    # = 9 flow conditions

    shallow_flows = [
        5.0,
        10.0,
        20.0,
    ]

    deep_flows = [
        5.0,
        10.0,
        20.0,
    ]

    result = (
        generate_human_dataset(
            human=human,
            shallow_flows=(
                shallow_flows
            ),
            deep_flows=(
                deep_flows
            ),
            nphoton=10_000_000,
            export=True,
            filename=(
                "human_dcs_dataset_test.csv"
            ),
        )
    )

    dataset = result[
        "dataset"
    ]

    print(
        "\n=== Final Summary ==="
    )

    print(
        f"Sex: "
        f"{result['geometry'].sex}"
    )

    print(
        f"Skin thickness: "
        f"{result['geometry'].skin_thickness_mm:.2f} mm"
    )

    print(
        f"Skin source: "
        f"{result['geometry'].skin_thickness_source}"
    )

    print(
        f"SAT thickness: "
        f"{result['geometry'].sat_thickness_mm:.2f} mm"
    )

    print(
        f"Unique shallow flows: "
        f"{sorted(dataset['flow_shallow'].unique())}"
    )

    print(
        f"Unique deep flows: "
        f"{sorted(dataset['flow_deep'].unique())}"
    )

    print(
        f"Unique SDS: "
        f"{sorted(dataset['sds_cm'].unique())}"
    )

    print(
        f"Tau points per curve: "
        f"{len(TAU_US)}"
    )

    expected_curves = (
        len(shallow_flows)
        * len(deep_flows)
        * 3
    )

    print(
        f"Total g2 curves: "
        f"{expected_curves}"
    )


if __name__ == "__main__":
    main()