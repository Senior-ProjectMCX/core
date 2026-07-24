"""
plot_comparison.py
------------------
Comprehensive comparison of simulated vs. physical DCS g2 curves.

Produces a multi-panel figure:
  Panel A (top)    : g2 curves for 9 representative flow combinations (3x3 grid)
  Panel B (mid-L)  : MAE per trial for each SDS (line plot)
  Panel C (mid-R)  : MAE & RMSE bar chart per SDS
  Panel D (bottom) : Scatter – simulated vs. physical g2 values (all trials, 3 SDS)

Usage:
  python code/plot_comparison.py \
      --physical  data/Dataset_phantom-intra40.mat \
      --simulated data/Dataset_simulated-intra40.mat \
      --output    plots/full_comparison.png
"""

import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── colour palette ────────────────────────────────────────────────────────────
SDS_COLORS  = ['#4C9BE8', '#F4845F', '#5DBE8A']   # 1 cm, 2 cm, 3 cm
SDS_LABELS  = ['1 cm SDS', '2 cm SDS', '3 cm SDS']
BG_DARK     = '#0F1117'
PANEL_BG    = '#1A1D27'
GRID_COLOR  = '#2E3347'
TEXT_COLOR  = '#E2E8F0'
ACCENT      = '#7C6AF7'


def load(path):
    mat = scipy.io.loadmat(path)
    g2  = mat['ave_gtau_intra40']     # (10, 501, 1540)
    ds  = mat['Dataset_set_intra40']  # (7,  1540)
    return g2, ds


def compute_metrics(phys_g2, sim_g2):
    """Returns (mae, rmse) arrays of shape (3, 1540) — one value per SDS per trial."""
    diff = phys_g2[:3] - sim_g2[:3]          # (3, 501, 1540)
    mae  = np.mean(np.abs(diff), axis=1)      # (3, 1540)
    rmse = np.sqrt(np.mean(diff**2, axis=1))  # (3, 1540)
    return mae, rmse


def pick_trials(phys_set, combos):
    """Find first trial index matching each (shallow_flow, deep_flow) combo."""
    indices = []
    labels  = []
    for fs, fd in combos:
        matches = np.where((phys_set[0] == fs) & (phys_set[1] == fd))[0]
        idx = int(matches[0]) if len(matches) > 0 else None
        indices.append(idx)
        labels.append(f"Qs={fs:.0f} Qd={fd:.0f} ml/min")
    return indices, labels


# ── main ──────────────────────────────────────────────────────────────────────
def main(physical_path, simulated_path, output_path):
    print("Loading datasets …")
    phys_g2, phys_set = load(physical_path)
    sim_g2,  _        = load(simulated_path)

    tau_us = np.linspace(0, 500, 501)   # delay axis in µs

    mae, rmse = compute_metrics(phys_g2, sim_g2)

    # 9 representative combos (3×3 grid)
    combos = [
        (0,  0),  (0,  20),  (0,  40),
        (20, 0),  (20, 20),  (20, 40),
        (40, 0),  (40, 20),  (40, 40),
    ]
    trial_idx, trial_labels = pick_trials(phys_set, combos)

    # ── figure layout ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family'     : 'DejaVu Sans',
        'font.size'       : 9,
        'text.color'      : TEXT_COLOR,
        'axes.facecolor'  : PANEL_BG,
        'axes.edgecolor'  : GRID_COLOR,
        'axes.labelcolor' : TEXT_COLOR,
        'xtick.color'     : TEXT_COLOR,
        'ytick.color'     : TEXT_COLOR,
        'figure.facecolor': BG_DARK,
        'grid.color'      : GRID_COLOR,
        'grid.linestyle'  : ':',
        'grid.alpha'      : 0.6,
    })

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        "DCS Simulation vs. Physical Dataset — Comprehensive Comparison",
        fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.995
    )

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              hspace=0.38,
                              height_ratios=[2.2, 1.0, 1.0])

    # ── Panel A : g2 curves (3×3 grid) ───────────────────────────────────────
    inner_A = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=outer[0], hspace=0.55, wspace=0.32
    )
    for i, (tidx, tlabel) in enumerate(zip(trial_idx, trial_labels)):
        ax = fig.add_subplot(inner_A[i // 3, i % 3])
        if tidx is None:
            ax.text(0.5, 0.5, "No trial\nfound", ha='center', va='center',
                    transform=ax.transAxes, color='#888')
            ax.set_title(tlabel, fontsize=8)
            continue
        for s in range(3):
            ax.plot(tau_us, phys_g2[s, :, tidx],
                    color=SDS_COLORS[s], lw=1.4, alpha=0.9,
                    label=f"Phys {SDS_LABELS[s]}" if i == 0 else "_")
            ax.plot(tau_us, sim_g2[s, :, tidx],
                    color=SDS_COLORS[s], lw=1.4, ls='--', alpha=0.9,
                    label=f"Sim {SDS_LABELS[s]}" if i == 0 else "_")
        ax.set_title(tlabel, fontsize=8, pad=4)
        ax.set_xlabel("τ (µs)", fontsize=7)
        ax.set_ylabel("g₂(τ)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True)

    # shared legend for Panel A
    legend_elements = []
    for s in range(3):
        legend_elements.append(Line2D([0], [0], color=SDS_COLORS[s], lw=1.4,
                                      label=f"Physical — {SDS_LABELS[s]}"))
        legend_elements.append(Line2D([0], [0], color=SDS_COLORS[s], lw=1.4,
                                      ls='--',
                                      label=f"Simulated — {SDS_LABELS[s]}"))
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.97), ncol=3,
               framealpha=0.15, fontsize=8, labelcolor=TEXT_COLOR,
               facecolor=PANEL_BG, edgecolor=GRID_COLOR)

    # ── Panel B : per-trial MAE (line) ────────────────────────────────────────
    inner_B = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], wspace=0.32
    )

    ax_b = fig.add_subplot(inner_B[0, 0])
    trial_x = np.arange(1, 1541)
    for s in range(3):
        ax_b.plot(trial_x, mae[s], color=SDS_COLORS[s],
                  lw=0.8, alpha=0.85, label=SDS_LABELS[s])
    ax_b.set_title("Per-Trial MAE (all 1540 trials)", fontsize=10)
    ax_b.set_xlabel("Trial index", fontsize=9)
    ax_b.set_ylabel("MAE", fontsize=9)
    ax_b.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG,
                edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax_b.grid(True)

    # ── Panel C : summary bar chart ───────────────────────────────────────────
    ax_c = fig.add_subplot(inner_B[0, 1])
    x      = np.arange(3)
    width  = 0.35
    mae_mean  = mae.mean(axis=1)
    rmse_mean = rmse.mean(axis=1)

    bars_mae  = ax_c.bar(x - width/2, mae_mean,  width, label='MAE',
                         color=SDS_COLORS, alpha=0.85, edgecolor=BG_DARK)
    bars_rmse = ax_c.bar(x + width/2, rmse_mean, width, label='RMSE',
                         color=SDS_COLORS, alpha=0.45, edgecolor=BG_DARK,
                         hatch='//')

    for bar, val in zip(bars_mae, mae_mean):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                  f"{val:.4f}", ha='center', va='bottom', fontsize=7,
                  color=TEXT_COLOR)
    for bar, val in zip(bars_rmse, rmse_mean):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                  f"{val:.4f}", ha='center', va='bottom', fontsize=7,
                  color=TEXT_COLOR)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(SDS_LABELS, fontsize=8)
    ax_c.set_title("Mean MAE & RMSE per SDS", fontsize=10)
    ax_c.set_ylabel("Error", fontsize=9)
    ax_c.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG,
                edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax_c.grid(True, axis='y')

    # ── Panel D : scatter – simulated vs physical ─────────────────────────────
    inner_D = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.30
    )
    for s in range(3):
        ax_d = fig.add_subplot(inner_D[0, s])
        # Subsample to 20k points to keep plot fast
        p_flat = phys_g2[s].flatten()
        q_flat = sim_g2[s].flatten()
        rng    = np.random.default_rng(0)
        idx    = rng.choice(len(p_flat), min(20000, len(p_flat)), replace=False)
        ax_d.scatter(p_flat[idx], q_flat[idx],
                     s=1.0, alpha=0.25, color=SDS_COLORS[s], rasterized=True)

        # identity line
        lo = min(p_flat.min(), q_flat.min())
        hi = max(p_flat.max(), q_flat.max())
        ax_d.plot([lo, hi], [lo, hi], '--', color='#FFD166', lw=1.5, label='y = x')

        r = np.corrcoef(p_flat, q_flat)[0, 1]
        ax_d.set_title(f"{SDS_LABELS[s]}\nR = {r:.4f}", fontsize=9)
        ax_d.set_xlabel("Physical g₂", fontsize=8)
        ax_d.set_ylabel("Simulated g₂", fontsize=8)
        ax_d.legend(fontsize=7, framealpha=0.2, facecolor=PANEL_BG,
                    edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        ax_d.grid(True)
        ax_d.tick_params(labelsize=7)

    # ── section labels ────────────────────────────────────────────────────────
    for label, y_pos in [("A  g₂ Curves — Representative Trials", 0.985),
                          ("B  Per-Trial Error / C  Summary Metrics",  0.635),
                          ("D  Scatter: Simulated vs. Physical",        0.325)]:
        fig.text(0.01, y_pos, label, fontsize=11, fontweight='bold',
                 color=ACCENT, va='top', transform=fig.transFigure)

    # ── save ──────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor=BG_DARK, edgecolor='none')
    print(f"[OK] Figure saved: {output_path}")

    # print summary table
    mae_mean  = mae.mean(axis=1)
    rmse_mean = rmse.mean(axis=1)
    print("\n================== Error Summary ==================")
    for s, lbl in enumerate(SDS_LABELS):
        print(f"  {lbl:10s}  MAE = {mae_mean[s]:.5f}  |  RMSE = {rmse_mean[s]:.5f}")
    print("===================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot comprehensive comparison of simulated vs. physical DCS g2 curves")
    parser.add_argument('--physical',   default='data/Dataset_phantom-intra40.mat')
    parser.add_argument('--simulated',  default='data/Dataset_simulated-intra40.mat')
    parser.add_argument('--output',     default='plots/full_comparison.png')
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args.physical, args.simulated, args.output)
