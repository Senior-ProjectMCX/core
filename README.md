# core — reference papers and phantom data

## docs/
- `boe-14-10-5358.pdf`, `Nakabayashi et al., Biomed. Opt. Express.pdf` — Nakabayashi et al. (2023),
  *Deep-learning-based separation of shallow and deep layer blood flow rates in diffuse correlation
  spectroscopy*, Biomed. Opt. Express 14(10) 5358
- `Wu et al., Neurophotonics 8(1), 015001 (2021).pdf` — Wu et al. (2021), Neurophotonics 8(1) 015001

## data/
- `metadata.csv`, `g2_SDS{1,2,3}cm.csv` — physical two-layer phantom measurements
  (flow presets/measured flow, BFI, g2(τ) for τ = 0–500 µs) used as ground truth
- `Dataset_metadata.csv`, `Dataset_flattened.csv`, `Dataset_*-intra40.mat`,
  `Dataset_simulated_SDS{1,2,3}cm.csv` — paper dataset (original .mat and CSV exports)
- `pathlengths.npz`, `pathlengths_realistic.npz`, `photon_database.npz` — earlier MCX photon outputs

The simulation code (Model A) is in https://github.com/Senior-ProjectMCX/wachirawit.chai
