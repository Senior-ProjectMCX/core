import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data_io
from dynamic_flow import calculate_g2, flow_to_alpha_db, DEFAULT_A
from generate_dataset import _downsample_detector


def _combo_split(fr_s, fr_d, test_frac=0.30, seed=0):
    """
    Split trials by unique (FR_set_shallow, FR_set_deep) combination into train/test so no
    combo leaks across the boundary. Returns (train_idx, test_idx) arrays of trial indices.
    """
    combos = {}
    for i, key in enumerate(zip(np.round(fr_s, 6), np.round(fr_d, 6))):
        combos.setdefault(key, []).append(i)
    keys = list(combos.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    test_keys = set(keys[:n_test])
    train_idx, test_idx = [], []
    for key, idxs in combos.items():
        (test_idx if key in test_keys else train_idx).extend(idxs)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), len(keys)


def _synth_curves(det, fr_s, fr_d, beta, tau, trials, a_s, a_d):
    """Synthetic g2 for the given trial indices. Returns array (3, N_TAU, len(trials))."""
    out = np.zeros((3, len(tau), len(trials)), dtype=np.float64)
    for j, i in enumerate(trials):
        aDB_s = flow_to_alpha_db(fr_s[i], a=a_s)
        aDB_d = flow_to_alpha_db(fr_d[i], a=a_d)
        for s in range(3):
            Ls, Ld, Ys, Yd = det[s]
            out[s, :, j] = calculate_g2(Ls, Ld, Ys, Yd, aDB_s, aDB_d, tau, beta[s])
    return out


def run_comparison(pathlengths_path, plot_output_path, metrics_output_path='plots/g2_metrics.png',
                   test_frac=0.30, seed=0, cal_max_photons=5000):
    meta = data_io.load_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    fr_s, fr_d = meta['FR_set_s'], meta['FR_set_d']

    pl = np.load(pathlengths_path)
    rng = np.random.default_rng(42)
    det = {s: _downsample_detector(pl, s + 1, cal_max_photons, rng) for s in range(3)}

    # --- Held-out split by flow combo (Action Item 4) ---
    train_idx, test_idx, n_combos = _combo_split(fr_s, fr_d, test_frac, seed)
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap!"
    print(f"{n_combos} unique flow combos -> {len(train_idx)} train / {len(test_idx)} test trials "
          f"(disjoint).")

    # --- Calibrate a SINGLE factor a on TRAIN combos only (one representative trial per combo) ---
    seen, cal_trials = set(), []
    for i in train_idx:
        key = (round(fr_s[i], 6), round(fr_d[i], 6))
        if key not in seen and (fr_s[i] > 0 or fr_d[i] > 0):   # skip the trivial zero-flow combo
            seen.add(key)
            cal_trials.append(i)
    cal_trials = np.array(cal_trials)

    def cal_loss(log_a):
        a = 10.0 ** log_a
        sim = _synth_curves(det, fr_s, fr_d, beta, tau, cal_trials, a, a)
        ref = phys_g2[:, :, cal_trials]
        return np.mean((sim - ref) ** 2)

    from scipy.optimize import minimize_scalar
    print(f"Calibrating single factor a on {len(cal_trials)} train combos...")
    res = minimize_scalar(cal_loss, bounds=(-9.0, -3.0), method='bounded')
    a_star = 10.0 ** res.x
    print(f"  calibrated a = {a_star:.4e} mm^2/s per (ml/min/100g)   (default was {DEFAULT_A:.1e})")

    # --- Report MAE/RMSE on TEST trials ONLY ---
    sim_test = _synth_curves(det, fr_s, fr_d, beta, tau, test_idx, a_star, a_star)
    ref_test = phys_g2[:, :, test_idx]
    mae = np.zeros(3)
    rmse = np.zeros(3)
    per_trial_mae = np.zeros((3, len(test_idx)))   # mean over tau, per SDS per test trial
    print("\n============ Held-out (TEST) curve metrics ============")
    for s in range(3):
        diff = sim_test[s] - ref_test[s]
        mae[s] = np.mean(np.abs(diff))
        rmse[s] = np.sqrt(np.mean(diff ** 2))
        per_trial_mae[s] = np.mean(np.abs(diff), axis=0)
        print(f"  SDS {data_io.SDS_LABELS[s]} - MAE: {mae[s]:.5f} | RMSE: {rmse[s]:.5f}")
    print("=======================================================")

    # --- Overlay plot (a few combos drawn from the TEST set) ---
    _overlay_plot(det, fr_s, fr_d, beta, tau, phys_g2, test_idx, a_star, plot_output_path)
    # --- MAE/RMSE metrics plot (held-out test set) ---
    _metrics_plot(mae, rmse, per_trial_mae, fr_s[test_idx] + fr_d[test_idx], metrics_output_path)
    return a_star


def _overlay_plot(det, fr_s, fr_d, beta, tau, phys_g2, test_idx, a_star, plot_output_path):
    # pick up to 4 test combos spanning low->high flow
    order = test_idx[np.argsort(fr_s[test_idx] + fr_d[test_idx])]
    picks = order[np.linspace(0, len(order) - 1, min(4, len(order))).astype(int)]
    tau_us = np.arange(data_io.N_TAU)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax_i, i in enumerate(picks):
        ax = axes[ax_i]
        sim = _synth_curves(det, fr_s, fr_d, beta, tau, [i], a_star, a_star)[:, :, 0]
        for s, lbl in enumerate(data_io.SDS_LABELS):
            ax.plot(tau_us, phys_g2[s, :, i], color=colors[s], ls='-', alpha=0.8, label=f"Phys {lbl}")
            ax.plot(tau_us, sim[s], color=colors[s], ls='--', lw=2, label=f"Sim {lbl}")
        ax.set_title(f"[TEST] Shallow={fr_s[i]:.0f}, Deep={fr_d[i]:.0f} ml/min/100g")
        ax.set_xlabel(r"Delay $\tau$ ($\mu$s)")
        ax.set_ylabel(r"$g_2(\tau)$")
        ax.grid(True, ls=':', alpha=0.6)
        if ax_i == 0:
            ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    out_dir = os.path.dirname(plot_output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(plot_output_path, dpi=200)
    print(f"Comparison plot saved to {plot_output_path}")


def _metrics_plot(mae, rmse, per_trial_mae, total_flow, metrics_output_path):
    """
    Two-panel held-out metrics figure:
      (A) grouped bar chart of MAE and RMSE per SDS,
      (B) per-trial MAE vs trials ordered by total flow (FR_s + FR_d), one line per SDS.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    order = np.argsort(total_flow)          # order test trials low -> high total flow
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: grouped bars MAE / RMSE per SDS
    x = np.arange(3)
    w = 0.38
    bA = axA.bar(x - w / 2, mae, w, label='MAE', color='#4c72b0')
    bB = axA.bar(x + w / 2, rmse, w, label='RMSE', color='#c44e52')
    axA.set_xticks(x)
    axA.set_xticklabels([f"{l} SDS" for l in data_io.SDS_LABELS])
    axA.set_ylabel('error (held-out test)')
    axA.set_title('(A) Held-out MAE / RMSE per SDS')
    axA.legend()
    axA.grid(True, axis='y', ls=':', alpha=0.6)
    for bars in (bA, bB):
        for r in bars:
            axA.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                         textcoords="offset points", xytext=(0, 2), ha='center', fontsize=8)

    # Panel B: per-trial MAE ordered by total flow
    for s, lbl in enumerate(data_io.SDS_LABELS):
        axB.plot(np.arange(len(order)), per_trial_mae[s, order],
                 color=colors[s], lw=1.2, alpha=0.85, label=f"{lbl} SDS")
    axB.set_xlabel('test trials ordered by total flow (low → high)')
    axB.set_ylabel('per-trial MAE')
    axB.set_title('(B) Per-trial MAE vs total flow')
    axB.legend()
    axB.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    out_dir = os.path.dirname(metrics_output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(metrics_output_path, dpi=200)
    print(f"Metrics plot saved to {metrics_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Held-out comparison of synthetic vs physical DCS g2")
    parser.add_argument('--pathlengths', type=str, default='data/pathlengths.npz')
    parser.add_argument('--output-plot', type=str, default='plots/g2_comparison.png')
    parser.add_argument('--metrics-plot', type=str, default='plots/g2_metrics.png')
    parser.add_argument('--test-frac', type=float, default=0.30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_comparison(args.pathlengths, args.output_plot, metrics_output_path=args.metrics_plot,
                   test_frac=args.test_frac, seed=args.seed)
