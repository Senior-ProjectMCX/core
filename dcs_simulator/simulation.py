"""
PMCX detector smoke test for DCS Simulator Core V1.

Supports:
1) Original reference phantom
2) Subject-specific human geometry

Human mode verifies:
Human input
    -> skin + SAT
    -> superficial depth
    -> voxel phantom
    -> MCX config

The actual PMCX human run is handled separately
by the human simulation pipeline.
"""

from pathlib import Path

import numpy as np
import pmcx

from dcs_processor import (
    calculate_g1,
    calculate_g2,
)

from parameters import (
    NX,
    NY,
    VOXEL_SIZE_MM,
)

from phantom import (
    create_two_layer_phantom,
    create_mcx_properties,
    validate_phantom,
    get_layer_info,
    print_phantom_summary,
)


# ============================================================
# TEST SETTINGS
# ============================================================

NPHOTON_SMOKE_TEST = 10_000_000

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


# ============================================================
# DETECTORS
# ============================================================

DETECTOR_RADIUS_VOXELS = 3.0


def build_detector_positions():
    """
    Build detectors at SDS 10, 20, and 30 mm.

    voxel size = 0.5 mm

    10 mm -> 20 voxels
    20 mm -> 40 voxels
    30 mm -> 60 voxels
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
# BUILD CONFIG
# ============================================================

def create_test_config(volume=None):
    """
    Build MCX config.

    Parameters
    ----------
    volume : np.ndarray or None

        None:
            Use original reference phantom.

        np.ndarray:
            Use externally supplied geometry,
            e.g. a subject-specific human volume.
    """

    if volume is None:
        volume = create_two_layer_phantom()

    validate_phantom(volume)

    properties = create_mcx_properties()

    detectors = build_detector_positions()

    cfg = {
        "nphoton": NPHOTON_SMOKE_TEST,

        "vol": volume,
        "prop": properties,

        "srcpos": SOURCE_POS,
        "srcdir": SOURCE_DIR,

        "detpos": detectors,

        # Save detected photons
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
# RUN ORIGINAL REFERENCE / DETECTOR TEST
# ============================================================

def run_detector_test():
    """
    Run the original PMCX reference-phantom detector test.
    """

    print("=== PMCX Detector Smoke Test ===")

    cfg = create_test_config()

    print(
        f"Photons: "
        f"{cfg['nphoton']}"
    )

    print(
        f"Volume shape: "
        f"{cfg['vol'].shape}"
    )

    print(
        f"Voxel size: "
        f"{cfg['unitinmm']} mm"
    )

    print(
        "Source position: "
        f"{cfg['srcpos']}"
    )

    print(
        "Source direction: "
        f"{cfg['srcdir']}"
    )

    print("\nDetectors:")

    for i, detector in enumerate(
        cfg["detpos"],
        start=1,
    ):

        print(
            f"Detector {i}: "
            f"x={detector[0]}, "
            f"y={detector[1]}, "
            f"z={detector[2]}, "
            f"radius={detector[3]} voxels"
        )

    print("\nRunning PMCX...")

    result = pmcx.run(cfg)

    print("\nPMCX run: PASS")

    print(
        "Result keys:",
        list(result.keys()),
    )

    # --------------------------------------------------------
    # Flux
    # --------------------------------------------------------

    if "flux" in result:

        flux = result["flux"]

        print("\n--- Flux ---")

        print(
            "Shape:",
            flux.shape,
        )

        print(
            "Finite:",
            np.all(
                np.isfinite(flux)
            ),
        )

        print(
            "Min:",
            np.min(flux),
        )

        print(
            "Max:",
            np.max(flux),
        )

    # --------------------------------------------------------
    # Detected photon data
    # --------------------------------------------------------

    if "detp" in result:

        detp = result["detp"]

        output_dir = (
            Path(__file__).resolve().parent
            / "output"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        detp_file = (
            output_dir
            / "detp_core_v1.npy"
        )

        np.save(
            detp_file,
            detp,
        )

        print(
            f"\nDetected photon histories saved to:"
            f"\n{detp_file}"
        )

        print(
            "\n--- Detected photons ---"
        )

        print(
            "Detector photon type:",
            type(detp),
        )

        if isinstance(
            detp,
            dict,
        ):

            print(
                "detp keys:",
                list(detp.keys()),
            )

            for key, value in detp.items():

                if hasattr(
                    value,
                    "shape",
                ):

                    print(
                        f"{key}: "
                        f"shape={value.shape}, "
                        f"dtype={value.dtype}"
                    )

                else:

                    print(
                        f"{key}: "
                        f"type={type(value)}"
                    )

        else:

            if hasattr(
                detp,
                "shape",
            ):

                print(
                    "detp shape:",
                    detp.shape,
                )

                print(
                    "\nFirst 10 detected photons:"
                )

                print(
                    detp[:, :10]
                )

                print(
                    "\nRow summaries:"
                )

                for row in range(
                    detp.shape[0]
                ):

                    values = detp[row]

                    print(
                        f"Row {row}: "
                        f"min={values.min():.6f}, "
                        f"max={values.max():.6f}, "
                        f"mean={values.mean():.6f}"
                    )

                print(
                    "\nUnique values in Row 0:"
                )

                unique_values, counts = (
                    np.unique(
                        detp[0],
                        return_counts=True,
                    )
                )

                print(
                    "\n--- Pathlength by detector ---"
                )

                detector_ids = (
                    detp[0].astype(int)
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

                    shallow = (
                        shallow_path[mask]
                    )

                    deep = (
                        deep_path[mask]
                    )

                    total = (
                        shallow
                        + deep
                    )

                    reached_deep = (
                        deep > 0
                    )

                    print(
                        f"\nDetector "
                        f"{detector_id}"
                    )

                    print(
                        f"Photons: "
                        f"{mask.sum()}"
                    )

                    if mask.sum() > 0:

                        print(
                            f"Mean shallow path: "
                            f"{shallow.mean():.3f} mm"
                        )

                        print(
                            f"Mean deep path: "
                            f"{deep.mean():.3f} mm"
                        )

                        print(
                            f"Mean total path: "
                            f"{total.mean():.3f} mm"
                        )

                        print(
                            f"Photons reaching deep layer: "
                            f"{reached_deep.sum()} / "
                            f"{mask.sum()} "
                            f"("
                            f"{100 * reached_deep.mean():.2f}%"
                            f")"
                        )

                for value, count in zip(
                    unique_values,
                    counts,
                ):

                    print(
                        f"value={value:g} -> "
                        f"{count} photons"
                    )

    else:

        print(
            "\nNo 'detp' returned."
        )

    # ========================================================
    # DCS g1 / g2
    # ========================================================

    beta_by_detector = {
        1: 1.43009039 - 1.0,
        2: 1.50906565 - 1.0,
        3: 1.45281563 - 1.0,
    }

    g2_results = {}

    if "detp" in result:

        import matplotlib.pyplot as plt

        detp = result["detp"]

        tau_us = np.arange(
            0,
            501,
            dtype=float,
        )

        tau_seconds = (
            tau_us
            * 1e-6
        )

        detector_labels = {
            1: "SDS 1 cm",
            2: "SDS 2 cm",
            3: "SDS 3 cm",
        }

        g1_results = {}

        print(
            "\n=== g1 summary: All Detectors ==="
        )

        for detector_id in (
            1,
            2,
            3,
        ):

            g1 = calculate_g1(
                detp=detp,
                tau_seconds=tau_seconds,
                detector_id=detector_id,
            )

            g1_results[
                detector_id
            ] = g1

            print(
                f"\n"
                f"{detector_labels[detector_id]}"
            )

            print(
                f"g1(0 us):   "
                f"{g1[0]:.6f}"
            )

            print(
                f"g1(100 us): "
                f"{g1[100]:.6f}"
            )

            print(
                f"g1(250 us): "
                f"{g1[250]:.6f}"
            )

            print(
                f"g1(500 us): "
                f"{g1[500]:.6f}"
            )

            print(
                "Finite:",
                np.all(
                    np.isfinite(g1)
                ),
            )

            print(
                "Monotonic:",
                np.all(
                    np.diff(g1)
                    <= 1e-12
                ),
            )

        print(
            "\n=== Generated g2 summary ==="
        )

        for detector_id in (
            1,
            2,
            3,
        ):

            beta = (
                beta_by_detector[
                    detector_id
                ]
            )

            g2 = calculate_g2(
                g1=(
                    g1_results[
                        detector_id
                    ]
                ),
                beta=beta,
            )

            g2_results[
                detector_id
            ] = g2

            print(
                f"\n"
                f"{detector_labels[detector_id]}"
            )

            print(
                f"beta:       "
                f"{beta:.6f}"
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
                f"g2(250 us): "
                f"{g2[250]:.6f}"
            )

            print(
                f"g2(500 us): "
                f"{g2[500]:.6f}"
            )

            print(
                "Finite:",
                np.all(
                    np.isfinite(g2)
                ),
            )

        # ====================================================
        # PLOT g1
        # ====================================================

        plt.figure(
            figsize=(9, 6)
        )

        for detector_id in (
            1,
            2,
            3,
        ):

            plt.plot(
                tau_us,
                g1_results[
                    detector_id
                ],
                label=(
                    detector_labels[
                        detector_id
                    ]
                ),
            )

        plt.xlabel(
            "Delay time (µs)"
        )

        plt.ylabel(
            "Normalized field autocorrelation "
            "g₁(τ)"
        )

        plt.title(
            "Generated DCS g₁ Curves"
        )

        plt.xlim(
            0,
            500,
        )

        plt.ylim(
            0,
            1.05,
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        plt.show()

        # ====================================================
        # PLOT g2
        # ====================================================

        plt.figure(
            figsize=(9, 6)
        )

        for detector_id in (
            1,
            2,
            3,
        ):

            plt.plot(
                tau_us,
                g2_results[
                    detector_id
                ],
                label=(
                    detector_labels[
                        detector_id
                    ]
                ),
            )

        plt.xlabel(
            "Delay time (µs)"
        )

        plt.ylabel(
            "Intensity autocorrelation "
            "g₂(τ)"
        )

        plt.title(
            "Generated DCS g₂ Curves"
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

    return result


# ============================================================
# HUMAN -> MCX GEOMETRY SMOKE TEST
# ============================================================

def run_human_geometry_test():
    """
    Verify:

        Human input
            -> skin + SAT
            -> superficial depth
            -> subject-specific voxel volume
            -> MCX config

    This function intentionally DOES NOT run PMCX yet.
    """

    from human_model import (
        HumanInput,
        build_human_thigh_geometry,
        print_human_summary,
    )

    print(
        "=== Human -> MCX Geometry Smoke Test ==="
    )

    # --------------------------------------------------------
    # Human input
    #
    # No measured skin thickness is supplied here,
    # so the sex-based reference value will be used.
    # --------------------------------------------------------

    human = HumanInput(
        sex="Female",
        sat_thickness_mm=10.0,
    )

    # --------------------------------------------------------
    # Human anatomy model
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
    # Create human voxel volume
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

    # --------------------------------------------------------
    # Send human volume into MCX config
    # --------------------------------------------------------

    cfg = create_test_config(
        volume=human_volume
    )

    # --------------------------------------------------------
    # Verify same volume reached config
    # --------------------------------------------------------

    if not np.array_equal(
        cfg["vol"],
        human_volume,
    ):

        raise ValueError(
            "MCX config did not receive "
            "the human volume correctly."
        )

    info = get_layer_info(
        cfg["vol"]
    )

    print(
        "\n--- MCX received geometry ---"
    )

    print(
        f"Shallow voxels: "
        f"{info['shallow_voxels']}"
    )

    print(
        f"Shallow depth: "
        f"{info['shallow_depth_mm']:.2f} mm"
    )

    print(
        f"Deep voxels: "
        f"{info['deep_voxels']}"
    )

    print(
        f"Deep depth: "
        f"{info['deep_depth_mm']:.2f} mm"
    )

    # --------------------------------------------------------
    # Verify expected human depth survived voxelization
    # --------------------------------------------------------

    requested_depth = (
        geometry.superficial_thickness_mm
    )

    voxelized_depth = (
        info["shallow_depth_mm"]
    )

    voxelization_error = abs(
        requested_depth
        - voxelized_depth
    )

    print(
        f"\nRequested superficial depth: "
        f"{requested_depth:.2f} mm"
    )

    print(
        f"Voxelized superficial depth: "
        f"{voxelized_depth:.2f} mm"
    )

    print(
        f"Voxelization difference: "
        f"{voxelization_error:.2f} mm"
    )

    # Maximum expected quantization error should
    # remain within approximately one voxel.
    if voxelization_error > VOXEL_SIZE_MM:

        raise ValueError(
            "Human superficial depth changed "
            "too much during voxelization."
        )

    print(
        "\nHuman -> MCX geometry "
        "connection: PASS"
    )

    return cfg


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Human geometry connection test.
    #
    # This does NOT run the 10-million-photon
    # PMCX simulation.
    run_human_geometry_test()

    # Original reference simulation can still
    # be run manually:
    #
    # run_detector_test()