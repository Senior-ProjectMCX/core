import os
import argparse
import numpy as np

import data_io
from dynamic_flow import calculate_g2, flow_to_alpha_db, DEFAULT_A


def _downsample_detector(pl_data, det_idx, max_photons, rng):
    """Return aligned (Ls, Ld, Ys, Yd) for one detector, subsampled with a SHARED index."""
    Ls = pl_data[f'det_{det_idx}_Ls']
    Ld = pl_data[f'det_{det_idx}_Ld']
    Ys = pl_data[f'det_{det_idx}_Ys']
    Yd = pl_data[f'det_{det_idx}_Yd']
    if len(Ls) > max_photons:
        idx = rng.choice(len(Ls), max_photons, replace=False)
        Ls, Ld, Ys, Yd = Ls[idx], Ld[idx], Ys[idx], Yd[idx]
    return Ls, Ld, Ys, Yd


def generate_synthetic_dataset(pathlengths_path, output_path,
                               a_shallow=DEFAULT_A, a_deep=DEFAULT_A, max_photons=10000):
    """
    Build synthetic g2(r, tau) for every trial using Wu Eq.(5) (momentum transfer) and a single
    calibration factor per layer (a_shallow == a_deep by default -- Action Item 3).

    alpha*D_B = a * FR. Writes CSVs via data_io.save_synthetic (same layout as physical data).
    """
    print(f"Loading physical metadata + g2 (CSV)...")
    meta = data_io.load_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    print(f"  beta = {beta[0]:.3f} / {beta[1]:.3f} / {beta[2]:.3f}")
    print(f"  calibration: a_shallow={a_shallow:.3e}, a_deep={a_deep:.3e} mm^2/s per ml/min/100g")

    print(f"Loading MCX pathlengths + momentum from {pathlengths_path}...")
    pl_data = np.load(pathlengths_path)
    for req in ('det_1_Ys', 'det_1_Yd'):
        if req not in pl_data:
            raise KeyError(f"'{req}' missing from {pathlengths_path}; re-run simulation.py "
                           "with momentum transfer enabled (ismomentum=1, savedetflag 'dpm').")

    rng = np.random.default_rng(42)
    det = {s: _downsample_detector(pl_data, s + 1, max_photons, rng) for s in range(3)}
    for s in range(3):
        print(f"  SDS {data_io.SDS_LABELS[s]}: {len(det[s][0])} photons after downsample")

    tau = data_io.TAU_ARRAY
    n_trials = meta['n_trials']
    print(f"Generating synthetic g2 for {n_trials} trials...")

    # Drive with the preset pump flow rates, matching the physical experiment setting.
    fr_s, fr_d = meta['FR_set_s'], meta['FR_set_d']
    sim_g2 = np.zeros((3, data_io.N_TAU, n_trials), dtype=np.float32)
    for i in range(n_trials):
        aDB_s = flow_to_alpha_db(fr_s[i], a=a_shallow)
        aDB_d = flow_to_alpha_db(fr_d[i], a=a_deep)
        for s in range(3):
            Ls, Ld, Ys, Yd = det[s]
            sim_g2[s, :, i] = calculate_g2(Ls, Ld, Ys, Yd, aDB_s, aDB_d, tau, beta[s])
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{n_trials}")

    written = data_io.save_synthetic(output_path, sim_g2, meta)
    print("Synthetic dataset written:")
    for f in written:
        print(f"  {f}")
    return sim_g2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic DCS g2 from MCX pathlengths+momentum (Eq.5)")
    parser.add_argument('--pathlengths', type=str, default='data/pathlengths.npz')
    parser.add_argument('--output', type=str, default='data/Dataset_simulated.csv',
                        help="Stem for synthetic CSVs (writes <stem>_SDS{1,2,3}cm.csv)")
    parser.add_argument('--a-shallow', type=float, default=DEFAULT_A, help="Shallow calibration a (mm^2/s per ml/min/100g)")
    parser.add_argument('--a-deep', type=float, default=DEFAULT_A, help="Deep calibration a; keep == a-shallow until c2/c1 passes")
    parser.add_argument('--max-photons', type=int, default=10000)
    args = parser.parse_args()

    generate_synthetic_dataset(args.pathlengths, args.output,
                               a_shallow=args.a_shallow, a_deep=args.a_deep,
                               max_photons=args.max_photons)
