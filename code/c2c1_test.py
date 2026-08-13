"""
c2/c1 geometry-validation gate (Action Item 2) -- run BEFORE any calibration.

Rationale
---------
Nakabayashi et al. (2023) fit a single BFI per SDS from the semi-infinite homogeneous DCS
model, then regress it against the two-layer flow rates (Eq.4):

    BFI = c1 * FR_shallow + c2 * FR_deep + c3

The ratio c2/c1 = W2/W1 reflects the layer sensitivities that follow from geometry + optics
ALONE, and is INVARIANT to a single flow->D_B calibration scale a (it cancels in the ratio).
Targets recomputed from Nakabayashi Eq.(10)-(12):

    1 cm: 0.113/0.804 = 0.140     2 cm: 0.373/0.969 = 0.385     3 cm: 0.624/1.359 = 0.459

If our simulated g2 (via Wu Eq.5) reproduces these ratios, the geometry/optics are right.
If not, no amount of calibration should be trusted to fix it -- investigate geometry first.

This script:
  1. Ground-truth check: regress the physical BFI (metadata.csv) vs FR_meas -> should recover
     ~0.14/0.38/0.46, validating the ground-truth side and the regression itself.
  2. Fitter validation: re-fit BFI from the physical g2 curves with the semi-infinite model and
     confirm it tracks the metadata BFI (Pearson R).
  3. If --synthetic is given: fit BFI from the synthetic g2, regress, and report c2/c1 vs target.
"""
import os
import argparse
import numpy as np
from scipy.optimize import minimize_scalar

import data_io
from dynamic_flow import K0, MUA_DEFAULT

# --- Optical / geometry constants for the semi-infinite fit ---
MUSP = 0.481                 # reduced scattering, mm^-1 (4.81 cm^-1)
MUA = MUA_DEFAULT            # absorption, mm^-1 (0.056)
SDS_MM = np.array([10.0, 20.0, 30.0])          # source-detector separations in mm
TARGET_C2C1 = {0: 0.140, 1: 0.385, 2: 0.459}   # per SDS index, from Eq.10-12

Z0 = 1.0 / MUSP              # mm
ZB = 1.76 / MUSP            # extrapolated boundary (Wu, n=1.37), mm


def semi_infinite_g1(bfi, rho_mm, tau):
    """
    Normalized g1(tau) from the semi-infinite correlation-diffusion Green's function
    (Nakabayashi Eq.3 / Wu Eq.1-2). bfi = alpha*D_B in mm^2/s, rho in mm, tau in s.
    Uses <dr^2(tau)> = 6*bfi*tau. Normalized by G1(tau=0) so g1(0)=1.
    """
    r1 = np.sqrt(rho_mm ** 2 + Z0 ** 2)
    r2 = np.sqrt(rho_mm ** 2 + (Z0 + 2.0 * ZB) ** 2)

    def _G1(k):
        return np.exp(-k * r1) / r1 - np.exp(-k * r2) / r2

    # K(tau) = sqrt(3 mua musp + musp^2 k0^2 * 6 * bfi * tau);  static K at tau=0
    k_tau = np.sqrt(3.0 * MUA * MUSP + (MUSP ** 2) * (K0 ** 2) * 6.0 * bfi * tau)
    k_static = np.sqrt(3.0 * MUA * MUSP)
    return _G1(k_tau) / _G1(k_static)


def fit_bfi_curve(g2_meas, rho_mm, beta, tau, bounds_log10=(-9.0, 1.0)):
    """
    Fit a single BFI (mm^2/s) to one measured g2 curve by least squares in log10(BFI) space
    (BFI spans orders of magnitude). beta is held fixed to the measured coherence factor.
    Returns the fitted BFI.
    """
    def loss(log_bfi):
        g1 = semi_infinite_g1(10.0 ** log_bfi, rho_mm, tau)
        g2_model = 1.0 + beta * (g1 ** 2)
        return np.mean((g2_model - g2_meas) ** 2)

    res = minimize_scalar(loss, bounds=bounds_log10, method='bounded')
    return 10.0 ** res.x


def fit_bfi_all(g2, beta, tau):
    """Fit BFI for every (SDS, trial). g2 shape (3, N_TAU, n_trials) -> bfi (3, n_trials)."""
    n_sds, _, n_trials = g2.shape
    bfi = np.zeros((n_sds, n_trials))
    for s in range(n_sds):
        for i in range(n_trials):
            bfi[s, i] = fit_bfi_curve(g2[s, :, i], SDS_MM[s], beta[s], tau)
        print(f"  fitted BFI for SDS {data_io.SDS_LABELS[s]}: "
              f"median={np.median(bfi[s]):.3e} mm^2/s")
    return bfi


def regress_c2c1(bfi, fr_s, fr_d):
    """
    For each SDS, least-squares fit BFI = c1*FR_s + c2*FR_d + c3 and return per-SDS
    (c1, c2, c3, R2, c2/c1). bfi shape (3, n_trials).
    """
    A = np.column_stack([fr_s, fr_d, np.ones_like(fr_s)])
    out = []
    for s in range(bfi.shape[0]):
        y = bfi[s]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        c1, c2, c3 = coef
        resid = y - A @ coef
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        ratio = c2 / c1 if c1 != 0 else float('nan')
        out.append((c1, c2, c3, r2, ratio))
    return out


def _print_regression(title, results, reference=None, ref_label='target'):
    """reference: per-SDS dict/list of ratios to compare against (defaults to Nakabayashi target)."""
    ref = reference if reference is not None else TARGET_C2C1
    print(f"\n{title}")
    print(f"  {'SDS':<5} {'c1':>11} {'c2':>11} {'c3':>11} {'R2':>7} {'c2/c1':>9} {ref_label:>9} {'status':>8}")
    for s, (c1, c2, c3, r2, ratio) in enumerate(results):
        tgt = ref[s]
        ok = 'PASS' if abs(ratio - tgt) <= 0.06 else 'off'
        print(f"  {data_io.SDS_LABELS[s]:<5} {c1:>11.3e} {c2:>11.3e} {c3:>11.3e} "
              f"{r2:>7.3f} {ratio:>9.3f} {tgt:>9.3f} {ok:>8}")


def main():
    parser = argparse.ArgumentParser(description="c2/c1 geometry validation gate")
    parser.add_argument('--synthetic', type=str, default=None,
                        help="Synthetic g2 stem/path (e.g. data/Dataset_simulated.csv) to validate")
    parser.add_argument('--fit-physical', action='store_true',
                        help="Also re-fit BFI from physical g2 to validate the fitter vs metadata")
    args = parser.parse_args()

    meta = data_io.load_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    print(f"Loaded {meta['n_trials']} trials. Coherence factors beta = "
          f"{beta[0]:.3f} / {beta[1]:.3f} / {beta[2]:.3f}  (expect ~0.407/0.494/0.490)")

    # 1. Ground-truth check: regress the physical BFI (from metadata) vs measured flow rates.
    gt = regress_c2c1(meta['BFI'], meta['FR_meas_s'], meta['FR_meas_d'])
    _print_regression("[1] GROUND-TRUTH  (metadata BFI vs FR_meas) -- should recover targets:", gt)

    # 2. Fit BFI from the physical g2 with our semi-infinite model. This is BOTH a fitter
    #    validation AND the fair same-fitter reference for the synthetic gate (our fitter has a
    #    small systematic ratio bias vs the paper's original BFI extraction, which cancels when
    #    synthetic and physical are processed identically).
    phys_ratio = None
    if args.fit_physical or args.synthetic:
        print("\n[2] Fitting BFI from PHYSICAL g2 with the semi-infinite model...")
        bfi_phys = fit_bfi_all(phys_g2, beta, tau)
        for s in range(3):
            r = np.corrcoef(bfi_phys[s], meta['BFI'][s])[0, 1]
            print(f"  SDS {data_io.SDS_LABELS[s]}: Pearson R(fitted, metadata BFI) = {r:.3f}")
        res_phys = regress_c2c1(bfi_phys, meta['FR_meas_s'], meta['FR_meas_d'])
        _print_regression("    c2/c1 from fitted-physical BFI (same-fitter reference):", res_phys)
        phys_ratio = [r[4] for r in res_phys]

    # 3. Validate the synthetic pipeline if provided -- compare against the same-fitter reference.
    if args.synthetic:
        print(f"\n[3] Fitting BFI from SYNTHETIC g2 ({args.synthetic})...")
        sim_g2 = data_io.load_synthetic_g2(args.synthetic)
        bfi_sim = fit_bfi_all(sim_g2, beta, tau)
        res_sim = regress_c2c1(bfi_sim, meta['FR_meas_s'], meta['FR_meas_d'])
        _print_regression("    c2/c1 from SYNTHETIC g2 -- THE GEOMETRY GATE:", res_sim,
                          reference=phys_ratio, ref_label='phys-fit')
        print(f"\n  Paper target (metadata) c2/c1 : {[round(TARGET_C2C1[s],3) for s in range(3)]}")
        print(f"  Same-fitter physical c2/c1    : {[round(r,3) for r in phys_ratio]}")
        print(f"  Synthetic (ours) c2/c1        : {[round(r[4],3) for r in res_sim]}")
        print("\n  If synthetic c2/c1 is far from the physical (same-fitter) reference, investigate "
              "geometry (d1, g, n) before any calibration -- do NOT split the scale to force a match.")


if __name__ == '__main__':
    main()
