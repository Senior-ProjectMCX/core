import scipy.io
import pandas as pd
import numpy as np
import argparse
import os

def convert_mat_to_csv(mat_path, output_dir):
    print(f"Loading MATLAB dataset from {mat_path}...")
    mat = scipy.io.loadmat(mat_path)
    
    # 1. Extract metadata matrix (Dataset_set_intra40: 7 x 1540)
    dataset_set = mat['Dataset_set_intra40']
    
    # Define metadata column names
    meta_columns = [
        'Shallow_Flow_Preset_Pump',
        'Deep_Flow_Preset_Pump',
        'Shallow_Flow_Measured_Outlet',
        'Deep_Flow_Measured_Outlet',
        'BFI_1cm_SDS',
        'BFI_2cm_SDS',
        'BFI_3cm_SDS'
    ]
    
    # Convert to DataFrame (needs to be transposed because it is 7 x 1540 in MAT)
    df_meta = pd.DataFrame(dataset_set.T, columns=meta_columns)
    
    # Save Metadata CSV
    meta_csv_path = os.path.join(output_dir, "Dataset_metadata.csv")
    df_meta.to_csv(meta_csv_path, index=True, index_label='Trial_Index')
    print(f"Metadata table successfully saved to: {meta_csv_path} (Shape: {df_meta.shape})")
    
    # 2. Extract autocorrelation functions (ave_gtau_intra40: 10 x 501 x 1540)
    gtau = mat['ave_gtau_intra40']
    num_trials = gtau.shape[2]
    
    # Flatten g2 curves (1cm, 2cm, 3cm channels) for each trial
    print("Flattening autocorrelation curves...")
    g2_1cm = gtau[0, :, :].T  # Shape: (1540, 501)
    g2_2cm = gtau[1, :, :].T  # Shape: (1540, 501)
    g2_3cm = gtau[2, :, :].T  # Shape: (1540, 501)
    
    # Define columns for curves
    cols_1cm = [f'g2_1cm_tau_{i}' for i in range(501)]
    cols_2cm = [f'g2_2cm_tau_{i}' for i in range(501)]
    cols_3cm = [f'g2_3cm_tau_{i}' for i in range(501)]
    
    # Create DataFrames
    df_1cm = pd.DataFrame(g2_1cm, columns=cols_1cm)
    df_2cm = pd.DataFrame(g2_2cm, columns=cols_2cm)
    df_3cm = pd.DataFrame(g2_3cm, columns=cols_3cm)
    
    # Concatenate metadata and curves horizontally
    df_full = pd.concat([df_meta, df_1cm, df_2cm, df_3cm], axis=1)
    
    # Save Full Flattened CSV
    full_csv_path = os.path.join(output_dir, "Dataset_flattened.csv")
    df_full.to_csv(full_csv_path, index=True, index_label='Trial_Index')
    print(f"Full flattened dataset successfully saved to: {full_csv_path} (Shape: {df_full.shape})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DCS two-layer MAT dataset to CSV")
    parser.add_argument('--input', type=str, default='data/Dataset_phantom-intra40.mat', help="Path to input MAT file")
    parser.add_argument('--outdir', type=str, default='data', help="Directory to save output CSV files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    convert_mat_to_csv(args.input, args.outdir)
