import numpy as np

def calculate_g1(Ls, Ld, flow_s, flow_d, tau_array):
    """
    Calculates the normalized electric field autocorrelation function g1(tau)
    for a given set of pathlengths Ls and Ld (in mm) and flow rates (in ml/min).
    
    Parameters:
      Ls: numpy array of pathlengths in the shallow layer (mm)
      Ld: numpy array of pathlengths in the deep layer (mm)
      flow_s: shallow layer flow rate (ml/min)
      flow_d: deep layer flow rate (ml/min)
      tau_array: numpy array of delay times (seconds)
    """
    if len(Ls) == 0:
        return np.ones_like(tau_array)
        
    # Constants
    lambda_vac = 785e-6  # 785 nm in mm
    n_tissue = 1.37      # Refractive index of tissue
    k0 = 2 * np.pi * n_tissue / lambda_vac  # Wavenumber in mm^-1
    mus_p = 0.481        # Reduced scattering coefficient in mm^-1 (4.81 cm^-1)
    
    # Convert volumetric flow rate to diffusion coefficient DB:
    # 1 ml/min corresponds to alpha * DB = 1e-6 mm^2/s
    DB_s = flow_s * 1e-6
    DB_d = flow_d * 1e-6
    
    # Calculate decay rates for each photon path:
    # Gamma_i = 2 * k0^2 * mus_p * (DB_s * Ls_i + DB_d * Ld_i)
    # Exponent is -Gamma_i * tau
    coeff = 2.0 * (k0**2) * mus_p
    decay_rates = coeff * (DB_s * Ls + DB_d * Ld)  # Shape: (N_photons,)
    
    # Vectorized computation over delay times:
    # exponent shape: (N_photons, N_delay_times)
    exponent = -np.outer(decay_rates, tau_array)
    
    # Average across all detected photons to get g1(tau)
    g1 = np.mean(np.exp(exponent), axis=0)
    
    return g1

def calculate_g2(Ls, Ld, flow_s, flow_d, tau_array, beta):
    """
    Calculates the intensity autocorrelation function g2(tau) = 1 + beta * |g1(tau)|^2.
    """
    g1 = calculate_g1(Ls, Ld, flow_s, flow_d, tau_array)
    g2 = 1.0 + beta * (g1 ** 2)
    return g2
