import os
import pmcx
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def run_simulation(n_photons=10000000):
    print(f"Initializing MCX grid with {n_photons:,} photons...")
    # Voxel grid: 100 x 50 x 50.5 mm at 0.5 mm resolution -> 200 x 100 x 101 voxels
    Nx, Ny, Nz = 200, 100, 101
    vol = np.zeros([Nx, Ny, Nz], dtype='uint8')
    
    # Z indices 0 to 10 (depth 0 to 5.5 mm): Shallow Layer (media 1)
    vol[:, :, 0:11] = 1
    # Z indices 11 to 100 (depth 5.5 to 50.5 mm): Deep Layer (media 2)
    vol[:, :, 11:] = 2
    
    # Optical properties array: [mu_a, g, mu_s, n]
    # We set mu_a = 0.0 so we can apply absorption post-simulation.
    # mus = mus' / (1 - g). For mus' = 4.81 cm^-1 (0.481 mm^-1) and g = 0.9: mus = 4.81 mm^-1
    prop = np.array([
        [0.0, 0.0, 0.0, 1.0],      # Background (air)
        [0.0, 0.9, 4.81, 1.37],    # Shallow layer
        [0.0, 0.9, 4.81, 1.37]     # Deep layer
    ], dtype='float32')
    
    # Source position: middle of top surface in voxel units (100, 50, 0)
    # Detectors: at X = 120 (1 cm SDS), 140 (2 cm SDS), 160 (3 cm SDS); Y = 50; Z = 0
    # Radius = 2.0 voxels (1.0 mm)
    detpos = np.array([
        [120.0, 50.0, 0.0, 2.0],  # Detector 1: 1 cm SDS
        [140.0, 50.0, 0.0, 2.0],  # Detector 2: 2 cm SDS
        [160.0, 50.0, 0.0, 2.0]   # Detector 3: 3 cm SDS
    ], dtype='float32')
    
    cfg = {
        'nphoton': n_photons,
        'vol': vol,
        'prop': prop,
        'srcpos': [100.0, 50.0, 0.0],
        'srcdir': [0.0, 0.0, 1.0],
        'tstart': 0.0,
        'tend': 5e-9,
        'tstep': 5e-9,
        'detpos': detpos,
        'bc': 'aaraaa',          # Absorbing except Fresnel reflection on the top surface
        'autopilot': 1,
        'gpuid': 1,
        'savedetflag': 'dpsw',
        'unitinmm': 0.5
    }
    
    res = pmcx.mcxlab(cfg)
    detp = res['detp']
    
    # Extract photon data
    detid = detp['detid']
    ppath = detp['ppath']  # shape (N_detected, 2) in voxel units
    w0 = detp['w0']
    
    # Save database
    save_path = r"D:\RA\data\photon_database.npz"
    np.savez(save_path, detid=detid, ppath=ppath, w0=w0)
    print(f"Simulation completed! Saved detected photons database to {save_path}")
    print(f"Total detected photons: {len(detid)}")
    for i in range(3):
        n_det = np.sum(detid == (i + 1))
        print(f"  Detector {i+1} ({i+1} cm SDS): {n_det} photons")

if __name__ == "__main__":
    # Test with 1 million photons
    run_simulation(n_photons=1000000)
