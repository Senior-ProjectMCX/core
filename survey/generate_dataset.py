import os
import argparse
import numpy as np
import scipy.io
from dynamic_flow import calculate_g2

def load_physical_metadata(mat_path):
    mat = scipy.io.loadmat(mat_path)
    dataset_set = mat['Dataset_set_intra40']
    preset_flow_s = dataset_set[0, :]
    preset_flow_d = dataset_set[1, :]
    measured_flow_s = dataset_set[2, :]
    measured_flow_d = dataset_set[3, :]
    physical_bfi_1cm = dataset_set[4, :]
    physical_bfi_2cm = dataset_set[5, :]
    physical_bfi_3cm = dataset_set[6, :]
    return {
        'dataset_set': dataset_set,
        'preset_flow_s': preset_flow_s,
        'preset_flow_d': preset_flow_d,
        'measured_flow_s': measured_flow_s,
        'measured_flow_d': measured_flow_d,
        'physical_bfi_1cm': physical_bfi_1cm,
        'physical_bfi_2cm': physical_bfi_2cm,
        'physical_bfi_3cm': physical_bfi_3cm,
        'physical_g2': mat['ave_gtau_intra40']
    }

def fit_coherence_factors(physical_g2):
    beta_1cm = np.mean(physical_g2[0, 0, :]) - 1.0
    beta_2cm = np.mean(physical_g2[1, 0, :]) - 1.0
    beta_3cm = np.mean(physical_g2[2, 0, :]) - 1.0
    print(f"Fitted coherence factors (beta):")
    print(f"  1 cm SDS: {beta_1cm:.4f}")
    print(f"  2 cm SDS: {beta_2cm:.4f}")
    print(f"  3 cm SDS: {beta_3cm:.4f}")
    return beta_1cm, beta_2cm, beta_3cm

def generate_synthetic_dataset(pathlengths_path, mat_path, output_mat_path):
    print(f"Loading physical dataset from {mat_path}...")
    metadata = load_physical_metadata(mat_path)
    print(f"Loading simulated pathlengths from {pathlengths_path}...")
    pl_data = np.load(pathlengths_path)
    det_1_Ls, det_1_Ld = pl_data['det_1_Ls'], pl_data['det_1_Ld']
    det_2_Ls, det_2_Ld = pl_data['det_2_Ls'], pl_data['det_2_Ld']
    det_3_Ls, det_3_Ld = pl_data['det_3_Ls'], pl_data['det_3_Ld']
    beta_1, beta_2, beta_3 = fit_coherence_factors(metadata['physical_g2'])
    tau_array = np.linspace(0, 500e-6, 501)
    num_trials = metadata['dataset_set'].shape[1]
    print(f"Generating synthetic g2 curves for {num_trials} trials...")
    sim_g2 = np.zeros((3, 501, num_trials), dtype=np.float32)
    for i in range(num_trials):
        flow_s = metadata['preset_flow_s'][i]
        flow_d = metadata['preset_flow_d'][i]
        sim_g2[0, :, i] = calculate_g2(det_1_Ls, det_1_Ld, flow_s, flow_d, tau_array, beta_1)
        sim_g2[1, :, i] = calculate_g2(det_2_Ls, det_2_Ld, flow_s, flow_d, tau_array, beta_2)
        sim_g2[2, :, i] = calculate_g2(det_3_Ls, det_3_Ld, flow_s, flow_d, tau_array, beta_3)
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{num_trials} trials...")
    simulated_dataset_gtau = np.zeros((10, 501, num_trials), dtype=np.float32)
    simulated_dataset_gtau[0:3, :, :] = sim_g2
    for row_idx in range(7):
        simulated_dataset_gtau[row_idx + 3, :, :] = metadata['dataset_set'][row_idx, :]
    save_dict = {
        'Dataset_set_intra40': metadata['dataset_set'],
        'ave_gtau_intra40': simulated_dataset_gtau
    }
    scipy.io.savemat(output_mat_path, save_dict)
    print(f"Synthetic dataset saved successfully to {output_mat_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic DCS dataset from MCX pathlengths")
    parser.add_argument('--pathlengths', type=str, default='code/pathlengths.npz', help="Path to simulated pathlengths file")
    parser.add_argument('--physical-mat', type=str, default='Dataset_phantom-intra40.mat', help="Path to physical phantom MAT file")
    parser.add_argument('--output', type=str, default='code/Dataset_simulated-intra40.mat', help="Output simulated MAT file path")
    args = parser.parse_args()
    generate_synthetic_dataset(args.pathlengths, args.physical_mat, args.output)
