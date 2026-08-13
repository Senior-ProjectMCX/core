import numpy as np

# --- Physical constants (785 nm phantom, Nakabayashi 2.1 / Wu Table 2 calf) ---
LAMBDA_VAC_MM = 785e-6      # 785 nm expressed in mm
N_TISSUE = 1.37            # refractive index of the phantom fluid
K0 = 2.0 * np.pi * N_TISSUE / LAMBDA_VAC_MM   # wavenumber k0 in mm^-1
MUA_DEFAULT = 0.056        # absorption coefficient, mm^-1 (0.56 cm^-1)

# Default flow -> Brownian diffusion calibration: alpha*D_B = a * FR  (Wu et al. Sec 2.4.3)
# a = 1e-6 mm^2/s per (ml/min/100g), SAME factor for both layers initially (Action Item 3).
DEFAULT_A = 1e-6


def flow_to_alpha_db(flow, a=DEFAULT_A, b=0.0):
    """Convert a volumetric flow rate (ml/min/100g) to alpha*D_B (mm^2/s): alpha*D_B = a*FR + b."""
    return a * flow + b


def calculate_g1(Ls, Ld, Ys, Yd, alpha_db_s, alpha_db_d, tau_array,
                 mua_s=MUA_DEFAULT, mua_d=MUA_DEFAULT):
    """
    Normalized electric-field autocorrelation g1(tau) using Wu et al. Neurophotonics
    8(1) 015001 (2021) Eq.(5) -- the momentum-transfer Monte Carlo form, NOT the
    diffusion approximation mu'_s * L.

        G1(tau) = (1/Np) * sum_n exp(-1/3 k0^2 sum_i Y_{n,i} <dr^2(tau)>_i)
                                * exp(-sum_i mua_i L_{n,i})

    With Brownian motion <dr^2(tau)>_i = 6 * (alpha*D_B)_i * tau (Wu Eq.3), the dynamic
    exponent per photon collapses to  2 * k0^2 * (aD_s * Y_s + aD_d * Y_d) * tau.
    g1 is normalized by the static (tau=0) absorption-weighted photon count so g1(0) = 1.

    Parameters
    ----------
    Ls, Ld : arrays of partial pathlength per photon in shallow / deep layer (mm)
    Ys, Yd : arrays of momentum transfer per photon in shallow / deep layer (dimensionless)
    alpha_db_s, alpha_db_d : alpha*D_B (mm^2/s) for the shallow / deep layer
    tau_array : delay times (seconds)
    mua_s, mua_d : absorption coefficients per layer (mm^-1)
    """
    if len(Ls) == 0:
        return np.ones_like(tau_array)

    # Static absorption weight per photon: exp(-(mua_s*Ls + mua_d*Ld))
    abs_w = np.exp(-(mua_s * Ls + mua_d * Ld))          # shape (Np,)

    # Per-photon dynamic decay rate (units s^-1): 2 k0^2 (aD_s Y_s + aD_d Y_d)
    decay_rates = 2.0 * (K0 ** 2) * (alpha_db_s * Ys + alpha_db_d * Yd)   # shape (Np,)

    norm = np.sum(abs_w)
    if norm == 0:
        return np.ones_like(tau_array)

    # g1(tau) = sum_p abs_w * exp(-decay*tau) / sum_p abs_w ; loop over tau to bound memory
    g1 = np.empty_like(tau_array)
    for t_idx, tau in enumerate(tau_array):
        g1[t_idx] = np.sum(abs_w * np.exp(-decay_rates * tau)) / norm

    return g1


def calculate_g2(Ls, Ld, Ys, Yd, alpha_db_s, alpha_db_d, tau_array, beta,
                 mua_s=MUA_DEFAULT, mua_d=MUA_DEFAULT):
    """Intensity autocorrelation via the Siegert relation g2 = 1 + beta * |g1|^2 (Wu Eq.4)."""
    g1 = calculate_g1(Ls, Ld, Ys, Yd, alpha_db_s, alpha_db_d, tau_array,
                      mua_s=mua_s, mua_d=mua_d)
    return 1.0 + beta * (g1 ** 2)
