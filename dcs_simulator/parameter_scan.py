"""
Parameter landscape scan for DCS Simulator Core V1.

Purpose:
- Check whether alphaDb_shallow and alphaDb_deep
  are identifiable from the current model/data.
- Verify whether the fitted deep parameter reaches
  the lower bound because of a real error trend.

This is diagnostic only.
It does NOT modify final model parameters.
"""

import numpy as np

from calibration import (
    load_data,
    prepare_detector_data,
    calculate_generated_g2,
    calculate_metrics,
)


# ============================================================
# SEARCH GRID
# ============================================================

# Wider than the previous optimizer lower bound.
# Log-spaced because alpha*Db spans orders of magnitude.

SHALLOW_VALUES = np.logspace(
    -7,
    -5,
    31,
)

DEEP_VALUES = np.logspace(
    -10,
    -4,
    61,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== DCS Parameter Identifiability Scan ==="
    )

    detp, reference = load_data()

    detector_data = prepare_detector_data(
        detp
    )

    print(
        f"Shallow values: {len(SHALLOW_VALUES)}"
    )

    print(
        f"Deep values:    {len(DEEP_VALUES)}"
    )

    print(
        f"Total combinations: "
        f"{len(SHALLOW_VALUES) * len(DEEP_VALUES)}"
    )

    print(
        "\nScanning..."
    )

    # --------------------------------------------------------
    # Store combined RMSE
    # --------------------------------------------------------

    rmse_grid = np.empty(
        (
            len(SHALLOW_VALUES),
            len(DEEP_VALUES),
        ),
        dtype=float,
    )

    # --------------------------------------------------------
    # Grid search
    # --------------------------------------------------------

    for i, shallow in enumerate(
        SHALLOW_VALUES
    ):

        for j, deep in enumerate(
            DEEP_VALUES
        ):

            generated = (
                calculate_generated_g2(
                    detector_data=detector_data,
                    reference=reference,
                    alpha_db_shallow=shallow,
                    alpha_db_deep=deep,
                )
            )

            _, combined_rmse = (
                calculate_metrics(
                    generated,
                    reference,
                )
            )

            rmse_grid[i, j] = (
                combined_rmse
            )

    # --------------------------------------------------------
    # Global minimum
    # --------------------------------------------------------

    min_index = np.unravel_index(
        np.argmin(rmse_grid),
        rmse_grid.shape,
    )

    best_i, best_j = min_index

    best_shallow = (
        SHALLOW_VALUES[best_i]
    )

    best_deep = (
        DEEP_VALUES[best_j]
    )

    best_rmse = (
        rmse_grid[best_i, best_j]
    )

    print(
        "\n=== Grid Minimum ==="
    )

    print(
        f"alphaDb shallow = "
        f"{best_shallow:.8e} mm^2/s"
    )

    print(
        f"alphaDb deep    = "
        f"{best_deep:.8e} mm^2/s"
    )

    print(
        f"Combined RMSE   = "
        f"{best_rmse:.6f}"
    )

    # --------------------------------------------------------
    # Check whether minimum hits search boundaries
    # --------------------------------------------------------

    print(
        "\n=== Boundary Check ==="
    )

    shallow_boundary = (
        best_i == 0
        or best_i == len(SHALLOW_VALUES) - 1
    )

    deep_boundary = (
        best_j == 0
        or best_j == len(DEEP_VALUES) - 1
    )

    print(
        "Shallow minimum on boundary:",
        shallow_boundary,
    )

    print(
        "Deep minimum on boundary:",
        deep_boundary,
    )

    # --------------------------------------------------------
    # Inspect deep direction near best shallow
    # --------------------------------------------------------

    print(
        "\n=== Deep Parameter Trend ==="
    )

    selected_indices = np.linspace(
        0,
        len(DEEP_VALUES) - 1,
        13,
        dtype=int,
    )

    for j in selected_indices:

        print(
            f"deep="
            f"{DEEP_VALUES[j]:.3e} "
            f"-> RMSE="
            f"{rmse_grid[best_i, j]:.6f}"
        )

    # --------------------------------------------------------
    # Inspect shallow direction near best deep
    # --------------------------------------------------------

    print(
        "\n=== Shallow Parameter Trend ==="
    )

    selected_indices = np.linspace(
        0,
        len(SHALLOW_VALUES) - 1,
        11,
        dtype=int,
    )

    for i in selected_indices:

        print(
            f"shallow="
            f"{SHALLOW_VALUES[i]:.3e} "
            f"-> RMSE="
            f"{rmse_grid[i, best_j]:.6f}"
        )

    # --------------------------------------------------------
    # Final interpretation helper
    # --------------------------------------------------------

    print(
        "\n=== Diagnostic ==="
    )

    if deep_boundary:

        print(
            "WARNING: Best deep parameter is on "
            "the scan boundary."
        )

        print(
            "Deep alphaDb is not yet reliably "
            "identified by the current model."
        )

    else:

        print(
            "Deep parameter has an interior minimum."
        )

    if shallow_boundary:

        print(
            "WARNING: Best shallow parameter is on "
            "the scan boundary."
        )

    else:

        print(
            "Shallow parameter has an interior minimum."
        )

    print(
        "\nParameter scan: PASS"
    )


if __name__ == "__main__":
    main()