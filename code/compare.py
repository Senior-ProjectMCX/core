import os
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def run_comparison(physical_mat_path, simulated_mat_path, plot_output_path):
    print(f"Loading physical dataset from {physical_mat_path}...")
    phys_mat = scipy.io.loadmat(physical_mat_path)
    phys_g2 = phys_mat['ave_gtau_intra40']  # Shape: (10, 501, 1540)
    phys_set = phys_mat['Dataset_set_intra40'] # Shape: (7, 1540)
    
    print(f"Loading simulated dataset from {simulated_mat_path}...")
    sim_mat = scipy.io.loadmat(simulated_mat_path)
    sim_g2 = sim_mat['ave_gtau_intra40']  # Shape: (10, 501, 1540)
    sim_set = sim_mat['Dataset_set_intra40'] # Shape: (7, 1540)
    
    # 1. Compute quantitative difference between physical and simulated curves
    # We compare rows 0, 1, 2 (the g2 curves for 1, 2, 3 cm SDS)
    mae_1cm = np.mean(np.abs(phys_g2[0, :, :] - sim_g2[0, :, :]))
    mae_2cm = np.mean(np.abs(phys_g2[1, :, :] - sim_g2[1, :, :]))
    mae_3cm = np.mean(np.abs(phys_g2[2, :, :] - sim_g2[2, :, :]))
    
    rmse_1cm = np.sqrt(np.mean((phys_g2[0, :, :] - sim_g2[0, :, :])**2))
    rmse_2cm = np.sqrt(np.mean((phys_g2[1, :, :] - sim_g2[1, :, :])**2))
    rmse_3cm = np.sqrt(np.mean((phys_g2[2, :, :] - sim_g2[2, :, :])**2))
    
    print("\n================== Curve Comparison Metrics ==================")
    print(f"1 cm SDS - MAE: {mae_1cm:.5f} | RMSE: {rmse_1cm:.5f}")
    print(f"2 cm SDS - MAE: {mae_2cm:.5f} | RMSE: {rmse_2cm:.5f}")
    print(f"3 cm SDS - MAE: {mae_3cm:.5f} | RMSE: {rmse_3cm:.5f}")
    print("==============================================================")
    
    # 2. Select a few representative combinations to plot
    # combinations = [(Shallow_flow, Deep_flow)]
    # Row 1 is preset shallow, Row 2 is preset deep
    combos_to_plot = [
        (0.0, 0.0),
        (5.0, 10.0),
        (20.0, 40.0),
        (40.0, 20.0)
    ]
    
    tau_array_us = np.arange(501) # 0 to 500 us
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (flow_s, flow_d) in enumerate(combos_to_plot):
        # Find indices matching this combination
        matches = np.where((phys_set[0, :] == flow_s) & (phys_set[1, :] == flow_d))[0]
        if len(matches) == 0:
            print(f"No trial found for combo (Shallow={flow_s}, Deep={flow_d}) in physical dataset.")
            continue
            
        trial_idx = matches[0] # Pick the first match
        
        ax = axes[idx]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green for 1, 2, 3 cm
        
        # Plot 1cm, 2cm, 3cm curves
        for sds_idx, sds_lbl in enumerate(['1 cm', '2 cm', '3 cm']):
            color = colors[sds_idx]
            # Physical: solid line with dots
            ax.plot(tau_array_us, phys_g2[sds_idx, :, trial_idx], label=f"Phys {sds_lbl}", color=color, linestyle='-', alpha=0.8)
            # Simulated: dashed line
            ax.plot(tau_array_us, sim_g2[sds_idx, :, trial_idx], label=f"Sim {sds_lbl}", color=color, linestyle='--', linewidth=2)
            
        ax.set_title(f"Flow: Shallow={flow_s:.1f}, Deep={flow_d:.1f} ml/min")
        ax.set_xlabel("Delay time $\tau$ ($\mu$s)")
        ax.set_ylabel("$g_2(\tau)$")
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='upper right')
            
    plt.tight_layout()
    
    plot_dir = os.path.dirname(plot_output_path)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.savefig(plot_output_path, dpi=300)
    print(f"Comparison plot saved successfully to {plot_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare physical and simulated DCS datasets")
    parser.add_argument('--physical', type=str, default='data/Dataset_phantom-intra40.mat', help="Path to physical MAT file")
    parser.add_argument('--simulated', type=str, default='data/Dataset_simulated-intra40.mat', help="Path to simulated MAT file")
    parser.add_argument('--output-plot', type=str, default='plots/g2_comparison.png', help="Path to save comparison plot image")
    
    args = parser.parse_args()
    
    run_comparison(args.physical, args.simulated, args.output_plot)
