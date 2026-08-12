from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# --------------------------------------------------
# Reference dataset
# --------------------------------------------------

REFERENCE_FILE = (
    Path(__file__).resolve().parent
    / "reference_data"
    / "Dataset_phantom-intra40.mat"
)

# ave_gtau channel mapping
G2_CHANNELS = {
    1: 0,  # SDS 1 cm
    2: 1,  # SDS 2 cm
    3: 2,  # SDS 3 cm
}


def load_reference_condition(shallow_flow, deep_flow):
    """
    Load all physical DCS measurements for one commanded
    shallow/deep flow condition.

    Returns:
        dict containing:
        - commanded flow
        - trial indices
        - measured flow
        - g2 curves
        - mean g2
        - standard deviation of g2
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference dataset not found:\n{REFERENCE_FILE}"
        )

    data = loadmat(REFERENCE_FILE)

    dataset = data["Dataset_set_intra40"]
    gtau = data["ave_gtau_intra40"]

    # --------------------------------------------------
    # Find requested condition
    # --------------------------------------------------

    condition_mask = (
        (dataset[0] == shallow_flow)
        & (dataset[1] == deep_flow)
    )

    trial_indices = np.where(condition_mask)[0]

    if len(trial_indices) == 0:
        raise ValueError(
            f"No reference condition found for "
            f"shallow={shallow_flow}, deep={deep_flow}"
        )

    # --------------------------------------------------
    # Extract measured flow
    # --------------------------------------------------

    measured_shallow = dataset[2, trial_indices]
    measured_deep = dataset[3, trial_indices]

    # --------------------------------------------------
    # Extract g2
    #
    # Result shape for each SDS:
    # (number_of_trials, 501)
    # --------------------------------------------------

    g2 = {}

    for sds_cm, channel in G2_CHANNELS.items():

        curves = gtau[channel, :, trial_indices]

        # NumPy advanced indexing may return (501, trials).
        # Force consistent shape: (trials, 501)
        if curves.shape[0] == 501:
            curves = curves.T

        g2[sds_cm] = {
            "trials": curves,
            "mean": np.mean(curves, axis=0),
            "std": np.std(curves, axis=0),
        }

    return {
        "commanded_shallow": shallow_flow,
        "commanded_deep": deep_flow,
        "trial_indices": trial_indices,
        "measured_shallow": measured_shallow,
        "measured_deep": measured_deep,
        "g2": g2,
    }

def plot_reference_g2(reference):
    """
    Plot mean physical g2 curves for SDS 1, 2, and 3 cm.
    """

    tau_us = np.arange(501)

    plt.figure(figsize=(9, 6))

    for sds_cm in (1, 2, 3):
        mean_g2 = reference["g2"][sds_cm]["mean"]

        plt.plot(
            tau_us,
            mean_g2,
            label=f"SDS {sds_cm} cm",
        )

    plt.xlabel("Delay time τ (μs)")
    plt.ylabel("g₂(τ)")
    plt.title(
        "Physical DCS Reference\n"
        "Shallow Flow = 5, Deep Flow = 20"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


def main():

    # Core V1 reference condition
    reference = load_reference_condition(
        shallow_flow=5,
        deep_flow=20,
    )

    print("=== DCS Physical Reference ===")

    print(
        f"Condition: "
        f"shallow={reference['commanded_shallow']}, "
        f"deep={reference['commanded_deep']}"
    )

    print(
        f"Number of trials: "
        f"{len(reference['trial_indices'])}"
    )

    print(
        f"Mean measured shallow flow: "
        f"{np.mean(reference['measured_shallow']):.4f}"
    )

    print(
        f"Mean measured deep flow: "
        f"{np.mean(reference['measured_deep']):.4f}"
    )

    print("\n=== g2 summary ===")

    for sds_cm in (1, 2, 3):

        g2_data = reference["g2"][sds_cm]

        print(f"\nSDS {sds_cm} cm")
        print("Trials shape:", g2_data["trials"].shape)
        print("Mean shape:", g2_data["mean"].shape)

        print(
            "Mean g2 first 5 points:",
            g2_data["mean"][:5],
        )

        print(
            "Mean g2 last 5 points:",
            g2_data["mean"][-5:],
        )

    plot_reference_g2(reference)


if __name__ == "__main__":
    main()