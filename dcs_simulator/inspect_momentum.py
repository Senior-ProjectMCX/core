"""
Inspect shallow/deep momentum transfer in detected photons.

Diagnostic only:
- does not modify model parameters
- does not rerun MCX
"""

import numpy as np

from calibration import (
    load_data,
    prepare_detector_data,
)


def summarize(name, values):
    values = np.asarray(values, dtype=float)

    print(f"{name}:")
    print(f"  mean   = {np.mean(values):.6f}")
    print(f"  median = {np.median(values):.6f}")
    print(f"  p25    = {np.percentile(values, 25):.6f}")
    print(f"  p75    = {np.percentile(values, 75):.6f}")
    print(f"  p95    = {np.percentile(values, 95):.6f}")
    print(f"  max    = {np.max(values):.6f}")


def main():

    print("=== DCS Momentum Diagnostic ===")

    detp, _ = load_data()

    detector_data = prepare_detector_data(detp)

    for detector_id in (1, 2, 3):

        data = detector_data[detector_id]

        shallow = data["momentum_shallow"]
        deep = data["momentum_deep"]
        weight = data["weight"]

        total = shallow + deep

        deep_fraction = np.divide(
            deep,
            total,
            out=np.zeros_like(deep),
            where=total > 0,
        )

        reaches_deep = deep > 0

        # Absorption-weighted momentum means
        weighted_shallow = np.sum(
            weight * shallow
        ) / np.sum(weight)

        weighted_deep = np.sum(
            weight * deep
        ) / np.sum(weight)

        weighted_total = (
            weighted_shallow
            + weighted_deep
        )

        print()
        print("=" * 50)
        print(f"SDS {detector_id} cm")
        print("=" * 50)

        print(f"Photons: {len(shallow)}")

        print(
            f"Photons with deep momentum > 0: "
            f"{np.sum(reaches_deep)} / {len(deep)} "
            f"({100 * np.mean(reaches_deep):.2f}%)"
        )

        print()

        summarize(
            "Shallow momentum",
            shallow,
        )

        print()

        summarize(
            "Deep momentum",
            deep,
        )

        print()

        summarize(
            "Deep fraction of total momentum",
            deep_fraction,
        )

        print()

        print("Absorption-weighted means:")
        print(
            f"  shallow = {weighted_shallow:.6f}"
        )
        print(
            f"  deep    = {weighted_deep:.6f}"
        )
        print(
            f"  total   = {weighted_total:.6f}"
        )

        if weighted_total > 0:
            print(
                f"  deep fraction = "
                f"{100 * weighted_deep / weighted_total:.2f}%"
            )

    print()
    print("Momentum diagnostic: PASS")


if __name__ == "__main__":
    main()