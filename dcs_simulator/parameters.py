"""
Core V1 parameters for the two-layer DCS phantom simulation.

Important:
- Physical/reference values and simulation assumptions are kept separate.
- Units used internally are primarily mm, s, and mm^-1.
- Detector radius and exact MCX coordinates are intentionally not defined yet.
"""

import numpy as np


# ============================================================
# REFERENCE CONDITION
# ============================================================

SHALLOW_FLOW_ML_MIN_100G = 5.0
DEEP_FLOW_ML_MIN_100G = 20.0


# ============================================================
# GEOMETRY
# ============================================================

VOXEL_SIZE_MM = 0.5

PHANTOM_X_MM = 100.0
PHANTOM_Y_MM = 50.0

SHALLOW_DEPTH_MM = 5.5
DEEP_DEPTH_MM = 35.0

TOTAL_DEPTH_MM = SHALLOW_DEPTH_MM + DEEP_DEPTH_MM

# Convert physical dimensions to voxel counts
NX = int(round(PHANTOM_X_MM / VOXEL_SIZE_MM))
NY = int(round(PHANTOM_Y_MM / VOXEL_SIZE_MM))

SHALLOW_VOXELS = int(round(SHALLOW_DEPTH_MM / VOXEL_SIZE_MM))
DEEP_VOXELS = int(round(DEEP_DEPTH_MM / VOXEL_SIZE_MM))

NZ = SHALLOW_VOXELS + DEEP_VOXELS


# ============================================================
# OPTICAL PROPERTIES
# ============================================================

WAVELENGTH_NM = 785.0

# Values reported for the physical phantom
MU_A_CM = 0.56
MU_SP_CM = 4.81

# Literature-supported assumptions for Core V1
ANISOTROPY_G = 0.89
REFRACTIVE_INDEX_N = 1.37

# Convert cm^-1 -> mm^-1
MU_A_MM = MU_A_CM / 10.0
MU_SP_MM = MU_SP_CM / 10.0

# Reduced scattering:
#
# mu_s' = mu_s * (1 - g)
#
# therefore:
#
# mu_s = mu_s' / (1 - g)

MU_S_MM = MU_SP_MM / (1.0 - ANISOTROPY_G)


# ============================================================
# SOURCE / DETECTOR GEOMETRY
# ============================================================

SDS_MM = np.array(
    [10.0, 20.0, 30.0],
    dtype=float,
)

# Exact source coordinates and detector radius will be defined
# when the PMCX configuration is implemented.


# ============================================================
# FLOW / DYNAMICS
# ============================================================

# Core V1 assumption:
#
# 1 ml/min/100g  <->  1e-6 mm^2/s alpha*Db

FLOW_TO_ALPHA_DB = 1e-6

ALPHA_DB_SHALLOW_MM2_S = (
    SHALLOW_FLOW_ML_MIN_100G * FLOW_TO_ALPHA_DB
)

ALPHA_DB_DEEP_MM2_S = (
    DEEP_FLOW_ML_MIN_100G * FLOW_TO_ALPHA_DB
)


# ============================================================
# DCS DELAY TIME
# ============================================================

TAU_US = np.arange(
    0.0,
    501.0,
    1.0,
)

# Convert microseconds -> seconds
TAU_S = TAU_US * 1e-6


# ============================================================
# SANITY CHECK
# ============================================================

def print_parameter_summary():
    print("=== DCS Simulator Core V1 Parameters ===")

    print("\n--- Reference condition ---")
    print(
        f"Shallow flow: "
        f"{SHALLOW_FLOW_ML_MIN_100G} ml/min/100g"
    )
    print(
        f"Deep flow: "
        f"{DEEP_FLOW_ML_MIN_100G} ml/min/100g"
    )

    print("\n--- Geometry ---")
    print(f"Voxel size: {VOXEL_SIZE_MM} mm")
    print(
        f"Domain: {NX} x {NY} x {NZ} voxels"
    )
    print(
        f"Physical size: "
        f"{PHANTOM_X_MM} x "
        f"{PHANTOM_Y_MM} x "
        f"{TOTAL_DEPTH_MM} mm"
    )
    print(
        f"Shallow: "
        f"{SHALLOW_DEPTH_MM} mm "
        f"({SHALLOW_VOXELS} voxels)"
    )
    print(
        f"Deep: "
        f"{DEEP_DEPTH_MM} mm "
        f"({DEEP_VOXELS} voxels)"
    )

    print("\n--- Optical properties ---")
    print(f"Wavelength: {WAVELENGTH_NM} nm")
    print(f"mu_a: {MU_A_MM:.6f} mm^-1")
    print(f"mu_s': {MU_SP_MM:.6f} mm^-1")
    print(f"mu_s: {MU_S_MM:.6f} mm^-1")
    print(f"g: {ANISOTROPY_G}")
    print(f"n: {REFRACTIVE_INDEX_N}")

    print("\n--- Source-detector separations ---")
    print(f"SDS: {SDS_MM} mm")

    print("\n--- Dynamics ---")
    print(
        f"alpha*Db shallow: "
        f"{ALPHA_DB_SHALLOW_MM2_S:.6e} mm^2/s"
    )
    print(
        f"alpha*Db deep: "
        f"{ALPHA_DB_DEEP_MM2_S:.6e} mm^2/s"
    )

    print("\n--- Delay time ---")
    print(
        f"tau: "
        f"{TAU_US[0]:.0f} - "
        f"{TAU_US[-1]:.0f} us"
    )
    print(f"Number of tau points: {len(TAU_US)}")


if __name__ == "__main__":
    print_parameter_summary()