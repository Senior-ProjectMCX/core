"""
Digital two-layer phantom for DCS Simulator Core V1.

Medium IDs:
1 = shallow layer
2 = deep layer

Modes
-----
Validation / reference phantom:
    Uses the original fixed shallow depth from parameters.py.

Human simulation:
    A subject-specific superficial depth can be supplied in mm.

The 0.05 mm Mylar separator in the physical phantom is
not explicitly modeled in Core V1.
"""

import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    NX,
    NY,
    NZ,
    SHALLOW_VOXELS,
    DEEP_VOXELS,
    VOXEL_SIZE_MM,
    MU_A_MM,
    MU_S_MM,
    ANISOTROPY_G,
    REFRACTIVE_INDEX_N,
)


# ============================================================
# MEDIUM IDs
# ============================================================

SHALLOW_MEDIUM_ID = 1
DEEP_MEDIUM_ID = 2


# ============================================================
# DEPTH / VOXEL CONVERSION
# ============================================================

def depth_mm_to_voxels(depth_mm):
    """
    Convert a physical depth in mm to a voxel count.

    The requested depth is rounded to the nearest voxel because
    the MCX geometry is discretized using VOXEL_SIZE_MM.
    """

    if depth_mm <= 0:
        raise ValueError(
            "Shallow/superficial depth must be greater than 0 mm."
        )

    voxels = int(round(depth_mm / VOXEL_SIZE_MM))

    # At least one shallow voxel must exist.
    voxels = max(1, voxels)

    if voxels >= NZ:
        raise ValueError(
            f"Requested shallow depth {depth_mm:.2f} mm "
            f"requires {voxels} voxels, but the domain depth "
            f"is only {NZ * VOXEL_SIZE_MM:.2f} mm."
        )

    return voxels


# ============================================================
# CREATE PHANTOM
# ============================================================

def create_two_layer_phantom(shallow_depth_mm=None):
    """
    Create a 3D two-layer voxel phantom.

    Parameters
    ----------
    shallow_depth_mm : float or None
        Physical thickness of the superficial/shallow region.

        None:
            Use the original reference phantom thickness from
            parameters.py (SHALLOW_VOXELS).

        float:
            Convert the supplied human superficial depth into
            voxels and use it as the shallow layer thickness.

    Returns
    -------
    volume : np.ndarray
        uint8 array containing medium IDs.
    """

    if shallow_depth_mm is None:
        shallow_voxels = SHALLOW_VOXELS
    else:
        shallow_voxels = depth_mm_to_voxels(
            shallow_depth_mm
        )

    volume = np.empty(
        (NX, NY, NZ),
        dtype=np.uint8,
    )

    # Shallow / superficial region
    volume[:, :, :shallow_voxels] = SHALLOW_MEDIUM_ID

    # Deep / muscle region
    volume[:, :, shallow_voxels:] = DEEP_MEDIUM_ID

    return volume


# ============================================================
# GET LAYER INFORMATION
# ============================================================

def get_layer_info(volume):
    """
    Determine layer geometry directly from a generated volume.

    This allows validation to work for both:
    - the original fixed phantom
    - subject-specific human geometry
    """

    if volume.shape != (NX, NY, NZ):
        raise ValueError(
            f"Incorrect phantom shape: "
            f"{volume.shape}, expected {(NX, NY, NZ)}"
        )

    # Inspect one vertical column.
    column = volume[0, 0, :]

    shallow_indices = np.where(
        column == SHALLOW_MEDIUM_ID
    )[0]

    if len(shallow_indices) == 0:
        raise ValueError(
            "No shallow layer found in volume."
        )

    shallow_voxels = len(shallow_indices)
    deep_voxels = NZ - shallow_voxels

    return {
        "shallow_voxels": shallow_voxels,
        "deep_voxels": deep_voxels,
        "shallow_depth_mm":
            shallow_voxels * VOXEL_SIZE_MM,
        "deep_depth_mm":
            deep_voxels * VOXEL_SIZE_MM,
    }


# ============================================================
# VALIDATE PHANTOM
# ============================================================

def validate_phantom(volume):
    """
    Perform basic sanity checks on either the fixed reference
    phantom or a subject-specific human phantom.
    """

    expected_shape = (NX, NY, NZ)

    if volume.shape != expected_shape:
        raise ValueError(
            f"Incorrect phantom shape: "
            f"{volume.shape}, expected {expected_shape}"
        )

    unique_media = np.unique(volume)

    expected_media = np.array(
        [
            SHALLOW_MEDIUM_ID,
            DEEP_MEDIUM_ID,
        ],
        dtype=np.uint8,
    )

    if not np.array_equal(
        unique_media,
        expected_media,
    ):
        raise ValueError(
            f"Unexpected medium IDs: {unique_media}"
        )

    layer_info = get_layer_info(volume)

    shallow_voxels = layer_info[
        "shallow_voxels"
    ]

    # Check shallow region
    if not np.all(
        volume[:, :, :shallow_voxels]
        == SHALLOW_MEDIUM_ID
    ):
        raise ValueError(
            "Shallow layer contains incorrect medium IDs."
        )

    # Check deep region
    if not np.all(
        volume[:, :, shallow_voxels:]
        == DEEP_MEDIUM_ID
    ):
        raise ValueError(
            "Deep layer contains incorrect medium IDs."
        )

    return True


# ============================================================
# MCX OPTICAL PROPERTY TABLE
# ============================================================

def create_mcx_properties():
    """
    Create the MCX optical property table.

    Columns:
        [mua, mus, g, n]

    V1 assumption:
        Human geometry may change, but the optical properties
        remain fixed using the current literature/reference
        values.
    """

    background = [
        0.0,
        0.0,
        1.0,
        1.0,
    ]

    shallow = [
        MU_A_MM,
        MU_S_MM,
        ANISOTROPY_G,
        REFRACTIVE_INDEX_N,
    ]

    deep = [
        MU_A_MM,
        MU_S_MM,
        ANISOTROPY_G,
        REFRACTIVE_INDEX_N,
    ]

    properties = np.array(
        [
            background,
            shallow,
            deep,
        ],
        dtype=float,
    )

    return properties


# ============================================================
# VISUALIZE PHANTOM
# ============================================================

def plot_phantom_cross_section(
    volume,
    title="Digital Two-Layer Phantom",
):
    """
    Plot an X-Z cross-section through the center
    of the phantom.
    """

    center_y = NY // 2

    cross_section = volume[
        :,
        center_y,
        :
    ].T

    plt.figure(figsize=(10, 5))

    plt.imshow(
        cross_section,
        origin="upper",
        aspect="auto",
        extent=[
            0,
            NX * VOXEL_SIZE_MM,
            NZ * VOXEL_SIZE_MM,
            0,
        ],
    )

    plt.xlabel("X (mm)")
    plt.ylabel("Depth Z (mm)")
    plt.title(title)

    plt.colorbar(
        label="Medium ID",
        ticks=[
            SHALLOW_MEDIUM_ID,
            DEEP_MEDIUM_ID,
        ],
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_phantom_summary(volume, name="Phantom"):
    """
    Print geometry information for a generated volume.
    """

    info = get_layer_info(volume)

    shallow_count = np.count_nonzero(
        volume == SHALLOW_MEDIUM_ID
    )

    deep_count = np.count_nonzero(
        volume == DEEP_MEDIUM_ID
    )

    print(f"\n=== {name} ===")

    print(f"Shape: {volume.shape}")

    print(
        f"Shallow: "
        f"{info['shallow_voxels']} voxels = "
        f"{info['shallow_depth_mm']:.2f} mm"
    )

    print(
        f"Deep: "
        f"{info['deep_voxels']} voxels = "
        f"{info['deep_depth_mm']:.2f} mm"
    )

    print(
        f"Shallow voxel count: {shallow_count}"
    )

    print(
        f"Deep voxel count: {deep_count}"
    )

    print(
        f"Total voxel count: {volume.size}"
    )


# ============================================================
# MAIN SANITY TEST
# ============================================================

def main():

    print("=== Phantom Geometry Smoke Test ===")

    # --------------------------------------------------------
    # TEST 1: Original reference phantom
    # --------------------------------------------------------

    reference_volume = create_two_layer_phantom()

    validate_phantom(reference_volume)

    print_phantom_summary(
        reference_volume,
        name="Reference Phantom",
    )

    reference_info = get_layer_info(
        reference_volume
    )

    if (
        reference_info["shallow_voxels"]
        != SHALLOW_VOXELS
    ):
        raise ValueError(
            "Reference phantom geometry changed unexpectedly."
        )

    print(
        "\nReference phantom validation: PASS"
    )

    # --------------------------------------------------------
    # TEST 2: Human geometry example
    # --------------------------------------------------------

    human_depth_mm = 11.55

    human_volume = create_two_layer_phantom(
        shallow_depth_mm=human_depth_mm
    )

    validate_phantom(human_volume)

    print_phantom_summary(
        human_volume,
        name="Human Example",
    )

    human_info = get_layer_info(
        human_volume
    )

    print(
        f"\nRequested human superficial depth: "
        f"{human_depth_mm:.2f} mm"
    )

    print(
        f"Voxelized human superficial depth: "
        f"{human_info['shallow_depth_mm']:.2f} mm"
    )

    print(
        "\nHuman phantom validation: PASS"
    )

    # --------------------------------------------------------
    # Optical properties
    # --------------------------------------------------------

    properties = create_mcx_properties()

    print(
        "\n--- MCX optical property table ---"
    )

    print("[mua, mus, g, n]")

    for medium_id, prop in enumerate(
        properties
    ):
        print(
            f"Medium {medium_id}: {prop}"
        )

    print(
        "\nAll phantom geometry tests: PASS"
    )

    # --------------------------------------------------------
    # Visualize human example
    # --------------------------------------------------------

    plot_phantom_cross_section(
        human_volume,
        title=(
            "Human Two-Layer Geometry "
            f"(superficial = "
            f"{human_info['shallow_depth_mm']:.2f} mm)"
        ),
    )


if __name__ == "__main__":
    main()