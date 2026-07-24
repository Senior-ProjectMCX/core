import os
import argparse
import numpy as np
import scipy.io
from dynamic_flow import calculate_g2

def load_physical_metadata(mat_path):
    """
    Loads flow rates and BFI parameters from the physical dataset.
    """
    mat = scipy.io.loadmat(mat_path)
    # Dataset_set_intra40 shape: (7, 1540)
    dataset_set = mat['Dataset_set_intra40']
    
    # Extract preset pump flow rates (Rows 1 & 2)
    preset_flow_s = dataset_set[0, :]
    preset_flow_d = dataset_set[1, :]
    
    # Extract measured outlet flow rates (Rows 3 & 4)
    measured_flow_s = dataset_set[2, :]
    measured_flow_d = dataset_set[3, :]
    
    # Extract physical BFI values (Rows 5, 6, 7)
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
        'physical_g2': mat['ave_gtau_intra40']  # Shape: (10, 501, 1540)
    }

def fit_coherence_factors(physical_g2):
    """
    Fits the coherence factors (beta) from the physical dataset at tau -> 0.
    Since g2(0) = 1 + beta, beta = g2(0) - 1.
    We compute the average g2(0) for each of the three source-detector separations.
    """
    # physical_g2 shape: (10, 501, 1540)
    # The first index of the 501 delay points represents tau = 0
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
    
    # Extract pathlengths for detectors 1, 2, 3
    det_1_Ls, det_1_Ld = pl_data['det_1_Ls'], pl_data['det_1_Ld']
    det_2_Ls, det_2_Ld = pl_data['det_2_Ls'], pl_data['det_2_Ld']
    det_3_Ls, det_3_Ld = pl_data['det_3_Ls'], pl_data['det_3_Ld']
    
    # Downsample to a maximum of 10,000 photons to accelerate calculations
    max_photons = 10000
    np.random.seed(42)
    
    if len(det_1_Ls) > max_photons:
        idx = np.random.choice(len(det_1_Ls), max_photons, replace=False)
        det_1_Ls, det_1_Ld = det_1_Ls[idx], det_1_Ld[idx]
        
    if len(det_2_Ls) > max_photons:
        idx = np.random.choice(len(det_2_Ls), max_photons, replace=False)
        det_2_Ls, det_2_Ld = det_2_Ls[idx], det_2_Ld[idx]
        
    if len(det_3_Ls) > max_photons:
        idx = np.random.choice(len(det_3_Ls), max_photons, replace=False)
        det_3_Ls, det_3_Ld = det_3_Ls[idx], det_3_Ld[idx]
    
    # Fit coherence factors from physical data
    beta_1, beta_2, beta_3 = fit_coherence_factors(metadata['physical_g2'])
    
    # Delay times array: 501 points from 0 to 500 us (steps of 1 us)
    tau_array = np.linspace(0, 500e-6, 501)
    
    num_trials = metadata['dataset_set'].shape[1] # 1540
    print(f"Generating synthetic g2 curves for {num_trials} trials...")
    
    # Allocate simulated g2 tensor: (3, 501, 1540)
    sim_g2 = np.zeros((3, 501, num_trials), dtype=np.float32)
    
    # We use the preset pump flow rates (metadata['preset_flow_s'] and metadata['preset_flow_d'])
    # to drive the simulation, matching the physical settings.
    for i in range(num_trials):
        flow_s = metadata['preset_flow_s'][i]
        flow_d = metadata['preset_flow_d'][i]
        
        # Calculate g2 for SDS 1, 2, 3 cm
        sim_g2[0, :, i] = calculate_g2(det_1_Ls, det_1_Ld, flow_s, flow_d, tau_array, beta_1)
        sim_g2[1, :, i] = calculate_g2(det_2_Ls, det_2_Ld, flow_s, flow_d, tau_array, beta_2)
        sim_g2[2, :, i] = calculate_g2(det_3_Ls, det_3_Ld, flow_s, flow_d, tau_array, beta_3)
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{num_trials} trials...")
            
    # Package into (10, 501, 1540) to mirror the physical dataset schema
    # Rows 0-2: Simulated g2 curves
    # Rows 3-9: Copy metadata and physical labels from the physical dataset
    simulated_dataset_gtau = np.zeros((10, 501, num_trials), dtype=np.float32)
    simulated_dataset_gtau[0:3, :, :] = sim_g2
    
    # Rows 4 to 10 (which are indices 3 to 9 in Python) mirror Dataset_set_intra40
    # Let's copy this metadata exactly
    for row_idx in range(7):
        # Broadcast 1D vector of shape (1540,) to (501, 1540) for row_idx + 3
        simulated_dataset_gtau[row_idx + 3, :, :] = metadata['dataset_set'][row_idx, :]
        
    # Save as .mat file
    save_dict = {
        'Dataset_set_intra40': metadata['dataset_set'],
        'ave_gtau_intra40': simulated_dataset_gtau
    }
    
    scipy.io.savemat(output_mat_path, save_dict)
    print(f"Synthetic dataset saved successfully to {output_mat_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic DCS dataset from MCX pathlengths")
    parser.add_argument('--pathlengths', type=str, default='data/pathlengths.npz', help="Path to simulated pathlengths file")
    parser.add_argument('--physical-mat', type=str, default='data/Dataset_phantom-intra40.mat', help="Path to physical phantom MAT file")
    parser.add_argument('--output', type=str, default='data/Dataset_simulated-intra40.mat', help="Output simulated MAT file path")
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.pathlengths, args.physical_mat, args.output)
