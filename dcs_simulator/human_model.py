"""
Human thigh parameter model for DCS simulation.

V1 scope
--------
Human inputs are used to define subject-specific
two-layer thigh geometry.

Required input:
- Sex
- Measured thigh SAT thickness

Optional input:
- Measured skin thickness

If measured skin thickness is not available,
a sex-based literature reference value is used.

IMPORTANT:
- SAT thickness is NOT estimated from BMI or sex.
- SAT should be measured at or near the DCS measurement site.
- Sex-based skin thickness is a reference fallback,
  not a prediction of the subject's true skin thickness.
- Optical properties are not modified by sex or SAT in V1.
"""

from dataclasses import dataclass
from typing import Optional

import math


# ============================================================
# SKIN THICKNESS REFERENCE VALUES
# ============================================================

# Literature-based mean thigh dermal thickness values [mm].
#
# These values are used only when subject-specific measured
# skin thickness is not available.
#
# They should be treated as reference/default values rather
# than exact predictions for an individual subject.

MALE_SKIN_THICKNESS_MM = 1.89
FEMALE_SKIN_THICKNESS_MM = 1.65


# ============================================================
# HUMAN INPUT
# ============================================================

@dataclass
class HumanInput:
    """
    Raw subject information supplied by the user.

    Parameters
    ----------
    sex : str
        Biological sex used only for selecting the
        fallback skin-thickness reference.

    sat_thickness_mm : float
        Measured subcutaneous adipose tissue (SAT)
        thickness at or near the DCS measurement site.

    skin_thickness_mm : float or None
        Optional measured skin thickness.

        If None, the model uses the sex-based
        literature reference value.
    """

    sex: str
    sat_thickness_mm: float
    skin_thickness_mm: Optional[float] = None


# ============================================================
# HUMAN GEOMETRY
# ============================================================

@dataclass
class HumanThighGeometry:
    """
    Geometry derived from human input.
    """

    sex: str

    skin_thickness_mm: float
    sat_thickness_mm: float

    superficial_thickness_mm: float

    skin_thickness_source: str


# ============================================================
# NORMALIZE SEX
# ============================================================

def normalize_sex(sex: str) -> str:
    """
    Normalize and validate the sex input.
    """

    normalized = str(sex).strip().lower()

    if normalized == "male":
        return "Male"

    if normalized == "female":
        return "Female"

    raise ValueError(
        "Sex must be 'Male' or 'Female'."
    )


# ============================================================
# VALIDATE HUMAN INPUT
# ============================================================

def validate_human_input(
    human: HumanInput,
) -> None:
    """
    Validate subject-specific input before geometry creation.
    """

    normalize_sex(human.sex)

    # --------------------------------------------------------
    # SAT thickness
    # --------------------------------------------------------

    try:
        sat = float(human.sat_thickness_mm)
    except (TypeError, ValueError):
        raise ValueError(
            "SAT thickness must be a valid number."
        )

    if not math.isfinite(sat):
        raise ValueError(
            "SAT thickness must be a finite number."
        )

    if sat <= 0:
        raise ValueError(
            "SAT thickness must be greater than 0 mm."
        )

    # --------------------------------------------------------
    # Optional measured skin thickness
    # --------------------------------------------------------

    if human.skin_thickness_mm is not None:

        try:
            skin = float(human.skin_thickness_mm)
        except (TypeError, ValueError):
            raise ValueError(
                "Skin thickness must be a valid number."
            )

        if not math.isfinite(skin):
            raise ValueError(
                "Skin thickness must be a finite number."
            )

        if skin <= 0:
            raise ValueError(
                "Skin thickness must be greater than 0 mm."
            )


# ============================================================
# GET SKIN THICKNESS
# ============================================================

def get_skin_thickness(
    human: HumanInput,
):
    """
    Determine the skin thickness used by the model.

    Priority
    --------
    1. Subject-specific measured skin thickness
    2. Sex-based literature reference value

    Returns
    -------
    skin_thickness_mm : float
    source : str
    """

    sex = normalize_sex(human.sex)

    # --------------------------------------------------------
    # Use measured value when available
    # --------------------------------------------------------

    if human.skin_thickness_mm is not None:

        return (
            float(human.skin_thickness_mm),
            "Measured",
        )

    # --------------------------------------------------------
    # Otherwise use literature reference
    # --------------------------------------------------------

    if sex == "Male":

        return (
            MALE_SKIN_THICKNESS_MM,
            "Sex-based reference",
        )

    return (
        FEMALE_SKIN_THICKNESS_MM,
        "Sex-based reference",
    )


# ============================================================
# BUILD HUMAN THIGH GEOMETRY
# ============================================================

def build_human_thigh_geometry(
    human: HumanInput,
) -> HumanThighGeometry:
    """
    Convert subject input into the two-layer geometry
    required by the Human DCS simulation.

    V1 geometry
    -----------
    Superficial layer:
        skin + SAT

    Deep layer:
        muscle

    Muscle begins immediately below the
    superficial region.
    """

    validate_human_input(human)

    sex = normalize_sex(human.sex)

    skin_thickness_mm, skin_source = (
        get_skin_thickness(human)
    )

    sat_thickness_mm = float(
        human.sat_thickness_mm
    )

    superficial_thickness_mm = (
        skin_thickness_mm
        + sat_thickness_mm
    )

    return HumanThighGeometry(
        sex=sex,
        skin_thickness_mm=skin_thickness_mm,
        sat_thickness_mm=sat_thickness_mm,
        superficial_thickness_mm=(
            superficial_thickness_mm
        ),
        skin_thickness_source=skin_source,
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_human_summary(
    geometry: HumanThighGeometry,
) -> None:
    """
    Print geometry for debugging and validation.
    """

    print("=== Human Thigh Model V1 ===")

    print(
        f"Sex:                 "
        f"{geometry.sex}"
    )

    print()

    print(
        f"Skin thickness:      "
        f"{geometry.skin_thickness_mm:.2f} mm"
    )

    print(
        f"Skin source:         "
        f"{geometry.skin_thickness_source}"
    )

    print(
        f"SAT thickness:       "
        f"{geometry.sat_thickness_mm:.2f} mm"
    )

    print(
        f"Superficial depth:   "
        f"{geometry.superficial_thickness_mm:.2f} mm"
    )

    print(
        f"Muscle begins at:    "
        f"{geometry.superficial_thickness_mm:.2f} mm"
    )


# ============================================================
# STANDALONE SMOKE TEST
# ============================================================

def main():

    print(
        "=== Human Model Smoke Test ==="
    )

    # --------------------------------------------------------
    # Test 1:
    # Female subject without measured skin thickness
    # --------------------------------------------------------

    print(
        "\n--- Test 1: Sex-based skin reference ---"
    )

    subject_reference = HumanInput(
        sex="Female",
        sat_thickness_mm=10.0,
    )

    geometry_reference = (
        build_human_thigh_geometry(
            subject_reference
        )
    )

    print_human_summary(
        geometry_reference
    )

    if (
        geometry_reference.skin_thickness_mm
        != FEMALE_SKIN_THICKNESS_MM
    ):
        raise RuntimeError(
            "Female skin reference test failed."
        )

    if (
        geometry_reference.skin_thickness_source
        != "Sex-based reference"
    ):
        raise RuntimeError(
            "Skin source test failed."
        )

    # --------------------------------------------------------
    # Test 2:
    # Subject with measured skin thickness
    # --------------------------------------------------------

    print(
        "\n--- Test 2: Measured skin override ---"
    )

    subject_measured = HumanInput(
        sex="Female",
        sat_thickness_mm=10.0,
        skin_thickness_mm=1.72,
    )

    geometry_measured = (
        build_human_thigh_geometry(
            subject_measured
        )
    )

    print_human_summary(
        geometry_measured
    )

    if (
        geometry_measured.skin_thickness_mm
        != 1.72
    ):
        raise RuntimeError(
            "Measured skin override failed."
        )

    if (
        geometry_measured.skin_thickness_source
        != "Measured"
    ):
        raise RuntimeError(
            "Measured skin source test failed."
        )

    print()
    print(
        "Human model smoke test: PASS"
    )


if __name__ == "__main__":
    main()