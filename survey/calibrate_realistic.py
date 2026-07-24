import os
import scipy.io
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def main():
    # 1. Load realistic pathlengths (already in mm, and absorption is simulated during runtime)
    pl_path = r"D:\RA\data\pathlengths_realistic.npz"
    pl_data = np.load(pl_path)
    
    # 2. Setup DCS parameters
    lambda_vac = 785e-6  # mm
    n_tissue = 1.37
    k0 = 2 * np.pi * n_tissue / lambda_vac
    mus_p = 0.481  # mm^-1
    exponent_factor = 2.0 * (k0**2) * mus_p
    
    # 3. Load physical dataset
    mat_path = r"D:\RA\data\Dataset_phantom-intra40.mat"
    mat_data = scipy.io.loadmat(mat_path)
    ave_gtau = mat_data['ave_gtau_intra40']
    set_intra = mat_data['Dataset_set_intra40']
    
    tau = np.linspace(0, 500e-6, 501)
    
    # 4. Extract physical betas and find calibration trials
    trial_0_0 = 0
    trial_10_20 = np.where((set_intra[0, :] == 10) & (set_intra[1, :] == 20))[0]
    trial_20_40 = np.where((set_intra[0, :] == 20) & (set_intra[1, :] == 40))[0]
    trial_60_60 = np.where((set_intra[0, :] == 60) & (set_intra[1, :] == 60))[0]
    
    betas_real = [
        np.mean(ave_gtau[0, 0, :]) - 1.0,
        np.mean(ave_gtau[1, 0, :]) - 1.0,
        np.mean(ave_gtau[2, 0, :]) - 1.0
    ]
    print(f"Physical betas: {betas_real}")
    
    def compute_sim_g1(sds_idx, Db_s, Db_d):
        Ls = pl_data[f'det_{sds_idx+1}_Ls']
        Ld = pl_data[f'det_{sds_idx+1}_Ld']
        
        if len(Ls) == 0:
            return np.ones_like(tau)
            
        phase_var = exponent_factor * (Db_s * Ls[:, None] + Db_d * Ld[:, None]) * tau[None, :]
        return np.mean(np.exp(-phase_var), axis=0)

    # We will fit scale_s and scale_d globally using Nelder-Mead on SDS 1cm, 2cm, 3cm
    cal_trials = []
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
            
            for sds_idx in [0, 1, 2]:
                y_real = ave_gtau[sds_idx, :, trial_idx]
                g1_sim = compute_sim_g1(sds_idx, Db_s, Db_d)
                y_sim = 1.0 + betas_real[sds_idx] * (g1_sim**2)
                loss += np.mean((y_real - y_sim)**2)
                count += 1
        return loss / count

    print("\nOptimizing scaling factors for realistic geometry...")
    res_fit = opt.minimize(loss_func, [1.0, 1.0], method='Nelder-Mead', bounds=((0.01, 10.0), (0.01, 10.0)))
    scale_s, scale_d = res_fit.x
    print(f"Optimal scaling factors:")
    print(f"  scale_s: {scale_s:.4f}")
    print(f"  scale_d: {scale_d:.4f}")
    print(f"  Success: {res_fit.success}")
    
    # 5. Evaluate MAE and RMSE with these scaling factors
    print("\nEvaluating curves for all 1540 trials...")
    num_trials = set_intra.shape[1]
    mae = np.zeros(3)
    rmse = np.zeros(3)
    
    for i in range(num_trials):
        flow_s = set_intra[0, i]
        flow_d = set_intra[1, i]
        Db_s = scale_s * flow_s * 1e-6
        Db_d = scale_d * flow_d * 1e-6
        
        for sds_idx in range(3):
            y_real = ave_gtau[sds_idx, :, i]
            g1_sim = compute_sim_g1(sds_idx, Db_s, Db_d)
            y_sim = 1.0 + betas_real[sds_idx] * (g1_sim**2)
            
            mae[sds_idx] += np.mean(np.abs(y_real - y_sim))
            rmse[sds_idx] += np.mean((y_real - y_sim)**2)
            
    mae /= num_trials
    rmse = np.sqrt(rmse / num_trials)
    
    print("\n================== Realistic Geometry Comparison Metrics ==================")
    print(f"1 cm SDS - MAE: {mae[0]:.5f} | RMSE: {rmse[0]:.5f}")
    print(f"2 cm SDS - MAE: {mae[1]:.5f} | RMSE: {rmse[1]:.5f}")
    print(f"3 cm SDS - MAE: {mae[2]:.5f} | RMSE: {rmse[2]:.5f}")
    print("==========================================================================")
    
    # 6. Save validation plots
    os.makedirs(r"D:\RA\code\plots", exist_ok=True)
    test_cases = [
        ("Rest (0,0)", trial_0_0, (0.0, 0.0)),
        ("Low-Moderate (20,40)", trial_20_40[0] if len(trial_20_40) > 0 else 100, (20.0, 40.0)),
        ("High (60,60)", trial_60_60[0] if len(trial_60_60) > 0 else 500, (60.0, 60.0))
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
            rmse_val = np.sqrt(np.mean((y_real - y_sim)**2))
            
            # Plot
            line, = plt.plot(tau * 1e6, y_real, '--', label=f"Real {sds_name} (RMSE: {rmse_val:.4f})")
            plt.plot(tau * 1e6, y_sim, '-', color=line.get_color(), label=f"Sim {sds_name}")
            
        plt.title(f"DCS Autocorrelation $g_2(\\tau)$ (Realistic) - {name} [Shallow: {flow_s}, Deep: {flow_d} ml/min]")
        plt.xlabel("Delay Time $\\tau$ ($\\mu$s)")
        plt.ylabel("$g_2(\\tau)$")
        plt.legend(frameon=True)
        plt.tight_layout()
        
        filename = f"D:\\RA\\plots\\realistic_val_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"  Saved plot: {filename}")

if __name__ == "__main__":
    main()
