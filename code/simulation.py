import os
import argparse
import numpy as np
import pmcx

def create_voxel_grid(geometry='realistic', unitinmm=0.5):
    """
    Creates the 3D voxel grid representation of the two-layer flow phantom.
    Voxel values:
      0: Air/Void (outside the phantom)
      1: Shallow layer liquid (dynamic scattering, index 1)
      2: Deep layer liquid (dynamic scattering, index 2)
      3: Mylar sheet separator (static, index 3)
      4: Acrylic chamber walls/bottom (static, index 4)
    """
    if geometry == 'simplified':
        # Simple two-layer infinite slab model (60 x 60 x 50 mm)
        nx = int(60 / unitinmm)
        ny = int(60 / unitinmm)
        nz = int(50 / unitinmm)
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)
        
        # Z boundary for shallow layer (5.5 mm)
        z_shallow_end = int(5.5 / unitinmm) # e.g. 11 voxels
        
        # Assign Shallow layer
        vol[:, :, :z_shallow_end] = 1
        # Assign Deep layer for the rest of the grid
        vol[:, :, z_shallow_end:] = 2
        
    elif geometry == 'realistic':
        # Realistic phantom model matching physical dimensions (100 x 50 x 50.5 mm)
        nx = int(100 / unitinmm)  # 200 voxels
        ny = int(50 / unitinmm)   # 100 voxels
        nz = int(50.5 / unitinmm) # 101 voxels
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)
        
        # Define wall thickness (2.5 mm -> 5 voxels)
        wall_voxels = int(2.5 / unitinmm)
        
        # Fill acrylic walls (Sides)
        vol[:wall_voxels, :, :] = 4
        vol[-wall_voxels:, :, :] = 4
        vol[:, :wall_voxels, :] = 4
        vol[:, -wall_voxels:, :] = 4
        
        # Fluid chamber dimensions inside the walls
        # X: [wall_voxels, nx - wall_voxels)
        # Y: [wall_voxels, ny - wall_voxels)
        # Z: [0, nz - bottom_wall_voxels)
        
        z_shallow_end = int(5.5 / unitinmm) # 11 voxels
        z_mylar_end = z_shallow_end + 1     # 12 voxels
        z_deep_end = z_mylar_end + int(35 / unitinmm) # 12 + 70 = 82 voxels
        
        # Shallow Layer fluid
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, :z_shallow_end] = 1
        # Mylar Sheet Separator (0.05 mm physical, modeled as 1 voxel at 0.5 mm grid)
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_shallow_end:z_mylar_end] = 3
        # Deep Layer fluid
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_mylar_end:z_deep_end] = 2
        # Bottom acrylic wall
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_deep_end:] = 4
        
    else:
        raise ValueError(f"Unknown geometry type: {geometry}")
        
    return vol

def run_simulation(nphoton=1e7, geometry='realistic', unitinmm=0.5, dry_run=False):
    print(f"Setting up {geometry} geometry voxel grid...")
    vol = create_voxel_grid(geometry, unitinmm)
    print(f"Voxel grid shape: {vol.shape}")
    
    # Baseline optical properties at 785 nm:
    # Index 0: background (Air)
    # Index 1: Shallow layer liquid (mua=0.056, mus'=0.481 -> mus=4.81, g=0.9, n=1.37)
    # Index 2: Deep layer liquid (mua=0.056, mus'=0.481 -> mus=4.81, g=0.9, n=1.37)
    # Index 3: Mylar separator (mua=0.01, mus=10.0, g=0.9, n=1.64)
    # Index 4: Acrylic wall (mua=0.005, mus=0.0, g=1.0, n=1.49)
    prop = [
        [0.0, 0.0, 1.0, 1.0],      # Air/Background
        [0.056, 4.81, 0.9, 1.37],  # Shallow fluid
        [0.056, 4.81, 0.9, 1.37],  # Deep fluid
        [0.01, 10.0, 0.9, 1.64],   # Mylar Sheet
        [0.005, 0.0, 1.0, 1.49]    # Acrylic resin wall
    ]
    
    # Define source position (centered at the top surface Z=0)
    nx, ny, nz = vol.shape
    srcpos = [nx / 2.0, ny / 2.0, 0.0]
    
    # Define detectors at 1, 2, and 3 cm SDS in X-direction
    # Separation in voxels: 10/unitinmm, 20/unitinmm, 30/unitinmm
    sds_voxels = [int(10 / unitinmm), int(20 / unitinmm), int(30 / unitinmm)]
    det_radius = 2.0 # 1 mm radius in voxels (if unitinmm = 0.5)
    
    detpos = []
    for sds in sds_voxels:
        detpos.append([srcpos[0] + sds, srcpos[1], 0.0, det_radius])
        
    cfg = {
        'nphoton': int(nphoton),
        'vol': vol,
        'prop': prop,
        'srcpos': srcpos,
        'srcdir': [0.0, 0.0, 1.0], # Light points downwards (into Z)
        'srctype': 'pencil',
        'detpos': detpos,
        'issavedet': 1,
        'savedetflag': 'dp', # Save detector ID ('d') and partial pathlength ('p')
        'maxdetphoton': 5000000,
        'unitinmm': unitinmm,
        'tstart': 0.0,
        'tend': 5e-9,
        'tstep': 5e-9,
        'bc': 'aaaaaa', # Absorbing boundary conditions on all sides
    }
    
    if dry_run:
        print("Dry run requested. Configuration is correct and pmcx check passed.")
        return None
        
    print(f"Launching simulation with {nphoton:.1e} photons on GPU...")
    res = pmcx.mcxlab(cfg)
    print("Simulation finished.")
    print(f"Runtime: {res['stat']['runtime']} ms")
    
    detp = res['detp']
    detid = detp['detid']
    ppath = detp['ppath'] # Shape: (N_detected_photons, N_media)
    
    # Process pathlengths
    processed_paths = {}
    for det_idx in range(1, len(detpos) + 1):
        idx = (detid == det_idx)
        # Extract pathlengths in Shallow (index 1) and Deep (index 2) layers
        # Convert pathlengths from grid units to millimeters
        Ls = ppath[idx, 1] * unitinmm
        Ld = ppath[idx, 2] * unitinmm
        
        print(f"Detector {det_idx} (SDS {det_idx} cm) captured {len(Ls)} photons.")
        processed_paths[f'det_{det_idx}_Ls'] = Ls.astype(np.float32)
        processed_paths[f'det_{det_idx}_Ld'] = Ld.astype(np.float32)
        
    return processed_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DCS Monte Carlo simulation using PMCX")
    parser.add_argument('--nphoton', type=float, default=1e7, help="Number of photons to simulate")
    parser.add_argument('--geometry', type=str, default='realistic', choices=['simplified', 'realistic'], help="Geometry type")
    parser.add_argument('--output', type=str, default='data/pathlengths.npz', help="Output file path")
    parser.add_argument('--dry-run', action='store_true', help="Check parameters without running simulation")
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    paths = run_simulation(nphoton=args.nphoton, geometry=args.geometry, dry_run=args.dry_run)
    
    if paths is not None:
        np.savez_compressed(args.output, **paths)
        print(f"Pathlength data saved to {args.output}")
