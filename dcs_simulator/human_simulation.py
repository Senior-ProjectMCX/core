"""
Human DCS end-to-end smoke test.

Pipeline:
Human input
    -> skin thickness + SAT
    -> superficial thickness
    -> human voxel geometry
    -> PMCX photon simulation
    -> detected-photon momentum transfer
    -> g1(tau)
    -> Siegert relation
    -> g2(tau)

Human Model V1:
- Sex is required.
- Measured thigh SAT thickness is required.
- Measured skin thickness is optional.
- If measured skin thickness is not supplied,
  a sex-based literature reference is used.

This is a FUNCTIONAL smoke test, not the final
calibrated human model.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pmcx

from human_model import (
    HumanInput,
    build_human_thigh_geometry,
    print_human_summary,
)

from phantom import (
    create_two_layer_phantom,
    create_mcx_properties,
    validate_phantom,
    get_layer_info,
    print_phantom_summary,
)

from dcs_processor import (
    calculate_g1,
    calculate_g2,
)

from parameters import (
    NX,
    NY,
    VOXEL_SIZE_MM,
)


# ============================================================
# SETTINGS
# ============================================================

NPHOTON_HUMAN_TEST = 10_000_000

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

# Temporary V1 beta values.
#
# These are carried over from the physical reference setup
# so that the end-to-end pipeline can be tested.
#
# They can later be replaced by calibrated /
# instrument-specific beta values.

BETA_BY_DETECTOR = {
    1: 1.43009039 - 1.0,
    2: 1.50906565 - 1.0,
    3: 1.45281563 - 1.0,
}

DETECTOR_LABELS = {
    1: "SDS 1 cm",
    2: "SDS 2 cm",
    3: "SDS 3 cm",
}


# ============================================================
# DETECTORS
# ============================================================

def build_detector_positions():
    """
    Build detectors at SDS 10, 20, and 30 mm.
    """

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
# HUMAN MCX CONFIG
# ============================================================

def create_human_config(
    human_volume,
):
    """
    Create PMCX configuration for a human thigh volume.
    """

    validate_phantom(
        human_volume
    )

    cfg = {
        "nphoton": NPHOTON_HUMAN_TEST,

        "vol": human_volume,

        "prop": create_mcx_properties(),

        "srcpos": SOURCE_POS,
        "srcdir": SOURCE_DIR,

        "detpos": build_detector_positions(),

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
# DCS PROCESSING
# ============================================================

def process_human_dcs(
    detp,
):
    """
    Convert detected-photon histories into
    g1 and g2 curves for all three SDS detectors.
    """

    if detp.ndim != 2:
        raise ValueError(
            f"Expected 2D detp, got {detp.shape}"
        )

    if detp.shape[0] != 5:
        raise ValueError(
            "Expected DPM output with 5 rows, "
            f"got {detp.shape[0]}"
        )

    detector_ids = (
        detp[0]
        .astype(int)
    )

    results = {}

    print(
        "\n=== Human g1 / g2 Summary ==="
    )

    for detector_id in (
        1,
        2,
        3,
    ):

        photon_count = int(
            np.sum(
                detector_ids
                == detector_id
            )
        )

        print(
            f"\n{DETECTOR_LABELS[detector_id]}"
        )

        print(
            f"Detected photons: "
            f"{photon_count}"
        )

        if photon_count == 0:
            raise ValueError(
                f"No photons detected at detector "
                f"{detector_id}"
            )

        g1 = calculate_g1(
            detp=detp,
            tau_seconds=TAU_SECONDS,
            detector_id=detector_id,
        )

        beta = (
            BETA_BY_DETECTOR[
                detector_id
            ]
        )

        g2 = calculate_g2(
            g1=g1,
            beta=beta,
        )

        results[
            detector_id
        ] = {
            "g1": g1,
            "g2": g2,
            "beta": beta,
            "photons": photon_count,
        }

        print(
            f"g1(0 us):   "
            f"{g1[0]:.6f}"
        )

        print(
            f"g1(100 us): "
            f"{g1[100]:.6f}"
        )

        print(
            f"g1(500 us): "
            f"{g1[500]:.6f}"
        )

        print(
            f"g2(0 us):   "
            f"{g2[0]:.6f}"
        )

        print(
            f"g2(100 us): "
            f"{g2[100]:.6f}"
        )

        print(
            f"g2(500 us): "
            f"{g2[500]:.6f}"
        )

        print(
            "g1 finite:",
            np.all(
                np.isfinite(g1)
            ),
        )

        print(
            "g2 finite:",
            np.all(
                np.isfinite(g2)
            ),
        )

        print(
            "g1 monotonic:",
            np.all(
                np.diff(g1)
                <= 1e-12
            ),
        )

    return results


# ============================================================
# HUMAN SIMULATION
# ============================================================

def run_human_simulation():
    """
    Run the complete Human DCS V1 smoke-test pipeline.
    """

    print(
        "=== Human DCS End-to-End Smoke Test ==="
    )

    # --------------------------------------------------------
    # 1. Human input
    #
    # This smoke test intentionally leaves measured skin
    # thickness empty.
    #
    # Therefore:
    # Female -> sex-based skin reference = 1.65 mm
    # SAT    -> measured value = 10.00 mm
    #
    # Expected superficial depth = 11.65 mm
    # --------------------------------------------------------

    human = HumanInput(
        sex="Female",
        sat_thickness_mm=10.0,
    )

    # --------------------------------------------------------
    # 2. Human anatomy
    # --------------------------------------------------------

    geometry = (
        build_human_thigh_geometry(
            human
        )
    )

    print_human_summary(
        geometry
    )

    # --------------------------------------------------------
    # 3. Human voxel geometry
    # --------------------------------------------------------

    human_volume = (
        create_two_layer_phantom(
            shallow_depth_mm=(
                geometry
                .superficial_thickness_mm
            )
        )
    )

    validate_phantom(
        human_volume
    )

    print_phantom_summary(
        human_volume,
        name="Human MCX Volume",
    )

    layer_info = (
        get_layer_info(
            human_volume
        )
    )

    # --------------------------------------------------------
    # Geometry / voxelization check
    # --------------------------------------------------------

    requested_depth = (
        geometry
        .superficial_thickness_mm
    )

    voxelized_depth = (
        layer_info[
            "shallow_depth_mm"
        ]
    )

    voxelization_difference = abs(
        requested_depth
        - voxelized_depth
    )

    print(
        "\n--- Human geometry check ---"
    )

    print(
        f"Requested superficial depth: "
        f"{requested_depth:.2f} mm"
    )

    print(
        f"Voxelized superficial depth: "
        f"{voxelized_depth:.2f} mm"
    )

    print(
        f"Voxelization difference: "
        f"{voxelization_difference:.2f} mm"
    )

    if (
        voxelization_difference
        > VOXEL_SIZE_MM
    ):

        raise ValueError(
            "Human superficial depth changed "
            "too much during voxelization."
        )

    # --------------------------------------------------------
    # 4. MCX config
    # --------------------------------------------------------

    cfg = create_human_config(
        human_volume
    )

    print(
        "\n--- Human simulation settings ---"
    )

    print(
        f"Photons: "
        f"{cfg['nphoton']}"
    )

    print(
        f"Voxel size: "
        f"{cfg['unitinmm']} mm"
    )

    print(
        f"Voxelized superficial depth: "
        f"{layer_info['shallow_depth_mm']:.2f} mm"
    )

    print(
        f"Muscle/deep region depth: "
        f"{layer_info['deep_depth_mm']:.2f} mm"
    )

    print(
        "\nRunning PMCX with human geometry..."
    )

    # --------------------------------------------------------
    # 5. PMCX
    # --------------------------------------------------------

    result = pmcx.run(
        cfg
    )

    print(
        "\nHuman PMCX run: PASS"
    )

    print(
        "Result keys:",
        list(
            result.keys()
        ),
    )

    if "detp" not in result:
        raise ValueError(
            "PMCX did not return detected-photon data."
        )

    detp = result[
        "detp"
    ]

    print(
        "\n--- Detected photon data ---"
    )

    print(
        "detp shape:",
        detp.shape,
    )

    detector_ids = (
        detp[0]
        .astype(int)
    )

    shallow_path = detp[1]
    deep_path = detp[2]

    for detector_id in (
        1,
        2,
        3,
    ):

        mask = (
            detector_ids
            == detector_id
        )

        count = int(
            np.sum(mask)
        )

        print(
            f"\n{DETECTOR_LABELS[detector_id]}"
        )

        print(
            f"Photons: {count}"
        )

        if count > 0:

            reached_deep = (
                deep_path[mask]
                > 0
            )

            print(
                f"Mean shallow path: "
                f"{np.mean(shallow_path[mask]):.3f} mm"
            )

            print(
                f"Mean deep path: "
                f"{np.mean(deep_path[mask]):.3f} mm"
            )

            print(
                f"Reached muscle/deep layer: "
                f"{np.sum(reached_deep)} / {count} "
                f"({100.0 * np.mean(reached_deep):.2f}%)"
            )

    # --------------------------------------------------------
    # 6. Save detected photons separately from reference run
    # --------------------------------------------------------

    output_dir = (
        Path(__file__)
        .resolve()
        .parent
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detp_file = (
        output_dir
        / "detp_human_v1.npy"
    )

    np.save(
        detp_file,
        detp,
    )

    print(
        f"\nHuman detected-photon histories saved to:"
        f"\n{detp_file}"
    )

    # --------------------------------------------------------
    # 7. Momentum -> g1 -> g2
    # --------------------------------------------------------

    dcs_results = (
        process_human_dcs(
            detp
        )
    )

    # --------------------------------------------------------
    # 8. Plot g2
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for detector_id in (
        1,
        2,
        3,
    ):

        plt.plot(
            TAU_US,
            dcs_results[
                detector_id
            ]["g2"],
            label=(
                DETECTOR_LABELS[
                    detector_id
                ]
            ),
        )

    plt.xlabel(
        "Delay time τ (µs)"
    )

    plt.ylabel(
        "Intensity autocorrelation g₂(τ)"
    )

    plt.title(
        "Human Geometry DCS g₂ "
        "(Functional Smoke Test)"
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

    print(
        "\n======================================"
    )

    print(
        "Human input -> geometry -> MCX "
        "-> momentum -> g1 -> g2: PASS"
    )

    print(
        "======================================"
    )

    return {
        "human": human,
        "geometry": geometry,
        "volume": human_volume,
        "mcx_result": result,
        "dcs_results": dcs_results,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_human_simulation()