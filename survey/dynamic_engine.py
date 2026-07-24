import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from simulate_dcs import run_simulation

# Set matplotlib style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_photon_data():
    db_path = r"D:\RA\data\photon_database.npz"
    # If the database doesn't exist or is small, let's run a 100M photon simulation
    if not os.path.exists(db_path):
        print("Photon database not found. Running high-statistics simulation...")
        run_simulation(n_photons=100000000)
    else:
        # Check size of the database. If it's from our 1M test run, let's run a 100M run to ensure good statistics for the 3cm SDS.
        data = np.load(db_path)
        if len(data['detid']) < 5000:
            print("Database has low statistics (from test run). Running 100M photon simulation...")
            run_simulation(n_photons=100000000)
            
    data = np.load(db_path)
    return data['detid'], data['ppath'], data['w0']

def main():
    # 1. Load photon data
    detid, ppath, w0 = load_photon_data()
    
    # Voxel size in mm
    unitinmm = 0.5
    
    # Convert pathlengths from voxel units to mm
    L_shallow = ppath[:, 0] * unitinmm
    L_deep = ppath[:, 1] * unitinmm
    L_total = L_shallow + L_deep
    
    # Post-simulation absorption weighting
    # mu_a = 0.056 mm^-1
    mu_a = 0.056
    w_abs = np.exp(-mu_a * L_total)
    
    # 2. Setup DCS parameters
    lambda_0 = 785e-6  # mm
    n = 1.37
    k0 = 2 * np.pi * n / lambda_0
    mus_prime = 0.481  # mm^-1
    
    # Factor in the exponent: 2 * k0^2 * mus_prime
    # Equation: exp( - 2 * k0^2 * mus_prime * (Db_s * L_s + Db_d * L_d) * tau )
    exponent_factor = 2.0 * (k0**2) * mus_prime
    print(f"DCS physical parameters:")
    print(f"  k0: {k0:.4f} mm^-1")
    print(f"  2 * k0^2 * mus_prime: {exponent_factor:.4e} mm^-2")
    
    # 3. Load physical dataset for comparison
    mat_path = r"D:\RA\data\Dataset_phantom-intra40.mat"
    mat_data = scipy.io.loadmat(mat_path)
    ave_gtau = mat_data['ave_gtau_intra40']
    set_intra = mat_data['Dataset_set_intra40']
    
    # Delay times: 0 to 500 us (501 points)
    tau = np.linspace(0, 500e-6, 501)
    
    # 4. Calibration & Fitting function
    # We want to find the optimal scaling factors for Db_s and Db_d to match the real dataset.
    # The paper defines: Db = flow_rate * 1e-6 mm^2/s
    # We will introduce a scaling parameter: Db_s = scale_s * flow_rate_s * 1e-6
    # and Db_d = scale_d * flow_rate_d * 1e-6
    
    # Let's extract a few curves from the dataset for fitting.
    # We will pick:
    # - Trial 0: flow_rates = (0, 0)
    # - A trial with moderate shallow and deep flow
    # - A trial with high deep flow
    
    # Let's find some trials with specific flow rates in set_intra:
    # Row 1 (index 0) = shallow preset
    # Row 2 (index 1) = deep preset
    
    print("\nFinding trials for calibration...")
    trial_0_0 = 0 # Preset (0, 0) is usually trial 0
    trial_10_20 = np.where((set_intra[0, :] == 10) & (set_intra[1, :] == 20))[0]
    trial_20_40 = np.where((set_intra[0, :] == 20) & (set_intra[1, :] == 40))[0]
    trial_60_60 = np.where((set_intra[0, :] == 60) & (set_intra[1, :] == 60))[0]
    
    print(f"  Preset (0,0) trial: {trial_0_0}")
    print(f"  Preset (10,20) trials: {trial_10_20}")
    print(f"  Preset (20,40) trials: {trial_20_40}")
    print(f"  Preset (60,60) trials: {trial_60_60}")
    
    # We will calibrate beta first using (0,0) flow. At (0,0), g1(tau) = 1.0, so g2(tau) = 1.0 + beta.
    # Let's check beta from the real dataset at (0,0) for each detector:
    betas_real = []
    for sds_idx in range(3):
        # We average the first few points of the curve where it should be 1 + beta
        betas_real.append(ave_gtau[sds_idx, 0, trial_0_0] - 1.0)
    print(f"  Extracted betas from real (0,0) trial: SDS 1cm={betas_real[0]:.4f}, SDS 2cm={betas_real[1]:.4f}, SDS 3cm={betas_real[2]:.4f}")
    
    # Let's define the simulated g1 calculation function
    def compute_sim_g1(sds_idx, Db_s, Db_d):
        # Filter photons for this detector (detid is 1-based, so detector 1 has detid = 1)
        mask = (detid == (sds_idx + 1))
        Ls = L_shallow[mask]
        Ld = L_deep[mask]
        w = w_abs[mask]
        
        if len(Ls) == 0:
            return np.ones_like(tau)
            
        # We calculate the average of exp(-exponent_factor * (Db_s * Ls + Db_d * Ld) * tau) weighted by w
        # Ls and Ld are in mm. Db_s and Db_d are in mm^2/s. tau is in seconds.
        # So the exponent is dimensionless.
        # To compute this efficiently in vector form:
        # Ls, Ld shape: (N,)
        # tau shape: (501,)
        # We compute exponent of shape (501, N) and sum over photons
        # Shape of Ls[:, None] * tau[None, :] is (N, 501)
        phase_var = exponent_factor * (Db_s * Ls[:, None] + Db_d * Ld[:, None]) * tau[None, :]
        g1_num = np.sum(w[:, None] * np.exp(-phase_var), axis=0)
        g1_den = np.sum(w)
        return g1_num / g1_den

    # Now let's optimize scale_s and scale_d to minimize RMSE between simulated and real curves
    # for a set of calibration trials.
    # Calibration trials to fit:
    cal_trials = []
    # Pick one index from each
    if len(trial_10_20) > 0: cal_trials.append(trial_10_20[0])
    if len(trial_20_40) > 0: cal_trials.append(trial_20_40[0])
    if len(trial_60_60) > 0: cal_trials.append(trial_60_60[0])
    
    def loss_func(scales):
        scale_s, scale_d = scales
        loss = 0.0
        count = 0
        for trial_idx in cal_trials:
            flow_s = set_intra[0, trial_idx]
            flow_d = set_intra[1, trial_idx]
            Db_s = scale_s * flow_s * 1e-6
            Db_d = scale_d * flow_d * 1e-6
            
            for sds_idx in [0, 1]:  # Fit using 1cm and 2cm SDS (which have high photon statistics)
                y_real = ave_gtau[sds_idx, :, trial_idx]
                g1_sim = compute_sim_g1(sds_idx, Db_s, Db_d)
                y_sim = 1.0 + betas_real[sds_idx] * (g1_sim**2)
                loss += np.mean((y_real - y_sim)**2)
                count += 1
        return loss / count

    print("\nRunning calibration fit for scaling factors...")
    res_fit = opt.minimize(loss_func, [1.0, 1.0], method='Nelder-Mead', bounds=((0.01, 10.0), (0.01, 10.0)))
    scale_s, scale_d = res_fit.x
    print(f"Calibration completed! Fitted scaling factors:")
    print(f"  Shallow layer scale_s: {scale_s:.4f}")
    print(f"  Deep layer scale_d: {scale_d:.4f}")
    print(f"  Optimization Success: {res_fit.success}")
    
    # 5. Validation and Plotting
    print("\nEvaluating and plotting validation results...")
    os.makedirs(r"D:\RA\code\plots", exist_ok=True)
    
    test_cases = [
        ("Rest (0,0)", trial_0_0, (0.0, 0.0)),
        ("Low-Moderate (10,20)", cal_trials[0] if len(cal_trials) > 0 else 100, (10.0, 20.0)),
        ("High (60,60)", cal_trials[2] if len(cal_trials) > 2 else 500, (60.0, 60.0))
    ]
    
    for name, trial_idx, (flow_s, flow_d) in test_cases:
        Db_s = scale_s * flow_s * 1e-6
        Db_d = scale_d * flow_d * 1e-6
        
        plt.figure(figsize=(10, 6))
        for sds_idx, sds_name in enumerate(["1 cm SDS", "2 cm SDS", "3 cm SDS"]):
            y_real = ave_gtau[sds_idx, :, trial_idx]
            g1_sim = compute_sim_g1(sds_idx, Db_s, Db_d)
            y_sim = 1.0 + betas_real[sds_idx] * (g1_sim**2)
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((y_real - y_sim)**2))
            
            # Plot
            line, = plt.plot(tau * 1e6, y_real, '--', label=f"Real {sds_name} (RMSE: {rmse:.4f})")
            plt.plot(tau * 1e6, y_sim, '-', color=line.get_color(), label=f"Sim {sds_name}")
            
        plt.title(f"DCS Autocorrelation $g_2(\\tau)$ - {name} [Shallow: {flow_s}, Deep: {flow_d} ml/min]")
        plt.xlabel("Delay Time $\\tau$ ($\\mu$s)")
        plt.ylabel("$g_2(\\tau)$")
        plt.legend(frameon=True)
        plt.tight_layout()
        
        filename = f"D:\\RA\\plots\\validation_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"  Saved plot: {filename}")

if __name__ == "__main__":
    main()
