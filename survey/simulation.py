import os
import argparse
import numpy as np
import pmcx

def create_voxel_grid(geometry='realistic', unitinmm=0.5):
    if geometry == 'simplified':
        nx = int(60 / unitinmm)
        ny = int(60 / unitinmm)
        nz = int(50 / unitinmm)
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)
        z_shallow_end = int(5.5 / unitinmm)
        vol[:, :, :z_shallow_end] = 1
        vol[:, :, z_shallow_end:] = 2
    elif geometry == 'realistic':
        nx = int(100 / unitinmm)
        ny = int(50 / unitinmm)
        nz = int(50.5 / unitinmm)
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)
        wall_voxels = int(2.5 / unitinmm)
        vol[:wall_voxels, :, :] = 4
        vol[-wall_voxels:, :, :] = 4
        vol[:, :wall_voxels, :] = 4
        vol[:, -wall_voxels:, :] = 4
        z_shallow_end = int(5.5 / unitinmm)
        z_mylar_end = z_shallow_end + 1
        z_deep_end = z_mylar_end + int(35 / unitinmm)
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, :z_shallow_end] = 1
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_shallow_end:z_mylar_end] = 3
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_mylar_end:z_deep_end] = 2
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_deep_end:] = 4
    else:
        raise ValueError(f"Unknown geometry type: {geometry}")
    return vol

def run_simulation(nphoton=1e7, geometry='realistic', unitinmm=0.5, dry_run=False):
    print(f"Setting up {geometry} geometry voxel grid...")
    vol = create_voxel_grid(geometry, unitinmm)
    print(f"Voxel grid shape: {vol.shape}")
    prop = [
        [0.0, 0.0, 1.0, 1.0],
        [0.056, 4.81, 0.9, 1.37],
        [0.056, 4.81, 0.9, 1.37],
        [0.01, 10.0, 0.9, 1.64],
        [0.005, 0.0, 1.0, 1.49]
    ]
    nx, ny, nz = vol.shape
    srcpos = [nx / 2.0, ny / 2.0, 0.0]
    sds_voxels = [int(10 / unitinmm), int(20 / unitinmm), int(30 / unitinmm)]
    det_radius = 2.0
    detpos = []
    for sds in sds_voxels:
        detpos.append([srcpos[0] + sds, srcpos[1], 0.0, det_radius])
    cfg = {
        'nphoton': int(nphoton),
        'vol': vol,
        'prop': prop,
        'srcpos': srcpos,
        'srcdir': [0.0, 0.0, 1.0],
        'srctype': 'pencil',
        'detpos': detpos,
        'issavedet': 1,
        'savedetflag': 'dp',
        'maxdetphoton': 5000000,
        'unitinmm': unitinmm,
        'tstart': 0.0,
        'tend': 5e-9,
        'tstep': 5e-9,
        'bc': 'aaaaaa',
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
    ppath = detp['ppath']
    processed_paths = {}
    for det_idx in range(1, len(detpos) + 1):
        idx = (detid == det_idx)
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
    parser.add_argument('--output', type=str, default='code/pathlengths.npz', help="Output file path")
    parser.add_argument('--dry-run', action='store_true', help="Check parameters without running simulation")
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    paths = run_simulation(nphoton=args.nphoton, geometry=args.geometry, dry_run=args.dry_run)
    if paths is not None:
        np.savez_compressed(args.output, **paths)
        print(f"Pathlength data saved to {args.output}")
