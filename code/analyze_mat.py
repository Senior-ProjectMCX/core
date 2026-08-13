import sys
sys.stdout.reconfigure(encoding='utf-8')
import scipy.io
import numpy as np

mat = scipy.io.loadmat('data/Dataset_phantom-intra40.mat')
ds = mat['Dataset_set_intra40']
g2 = mat['ave_gtau_intra40']

# Find all trials where pump preset = (0, 0)
zero_mask = (ds[0,:] == 0) & (ds[1,:] == 0)
zero_indices = np.where(zero_mask)[0]
print(f"Trials with preset flow (0,0): {len(zero_indices)}")
print(f"  Indices: {zero_indices[:20]}...")

tau_us = np.linspace(2.5, 500, 501)  # delay time in microseconds

# === g2 curves at flow=0 ===
print("\n=== g2(τ) at flow=(0,0) — SDS 1cm ===")
for i in zero_indices[:5]:
    print(f"  Trial {i}: g2(0)={g2[0,0,i]:.4f}, g2(50us)={g2[0,20,i]:.4f}, "
          f"g2(250us)={g2[0,100,i]:.4f}, g2(500us)={g2[0,-1,i]:.4f}")

print("\n=== g2(τ) at flow=(0,0) — SDS 3cm ===")
for i in zero_indices[:5]:
    print(f"  Trial {i}: g2(0)={g2[2,0,i]:.4f}, g2(50us)={g2[2,20,i]:.4f}, "
          f"g2(250us)={g2[2,100,i]:.4f}, g2(500us)={g2[2,-1,i]:.4f}")

# === Compare with high flow ===
high_mask = (ds[0,:] == 60) & (ds[1,:] == 60)
high_indices = np.where(high_mask)[0]
print(f"\n=== g2(τ) at flow=(60,60) — SDS 1cm ===")
for i in high_indices[:3]:
    print(f"  Trial {i}: g2(0)={g2[0,0,i]:.4f}, g2(50us)={g2[0,20,i]:.4f}, "
          f"g2(250us)={g2[0,100,i]:.4f}, g2(500us)={g2[0,-1,i]:.4f}")

# === Decay amount: how much does g2 drop from tau=0 to tau=500us ===
print("\n=== Decay amount (g2(0) - g2(500us)) ===")

# flow=0
decay_0_1cm = np.mean(g2[0, 0, zero_mask] - g2[0, -1, zero_mask])
decay_0_2cm = np.mean(g2[1, 0, zero_mask] - g2[1, -1, zero_mask])
decay_0_3cm = np.mean(g2[2, 0, zero_mask] - g2[2, -1, zero_mask])
print(f"  Flow=(0,0):  1cm={decay_0_1cm:.4f}, 2cm={decay_0_2cm:.4f}, 3cm={decay_0_3cm:.4f}")

# flow=5,5
f55_mask = (ds[0,:] == 5) & (ds[1,:] == 5)
decay_55_1cm = np.mean(g2[0, 0, f55_mask] - g2[0, -1, f55_mask])
decay_55_2cm = np.mean(g2[1, 0, f55_mask] - g2[1, -1, f55_mask])
decay_55_3cm = np.mean(g2[2, 0, f55_mask] - g2[2, -1, f55_mask])
print(f"  Flow=(5,5):  1cm={decay_55_1cm:.4f}, 2cm={decay_55_2cm:.4f}, 3cm={decay_55_3cm:.4f}")

# flow=60,60
decay_60_1cm = np.mean(g2[0, 0, high_mask] - g2[0, -1, high_mask])
decay_60_2cm = np.mean(g2[1, 0, high_mask] - g2[1, -1, high_mask])
decay_60_3cm = np.mean(g2[2, 0, high_mask] - g2[2, -1, high_mask])
print(f"  Flow=(60,60): 1cm={decay_60_1cm:.4f}, 2cm={decay_60_2cm:.4f}, 3cm={decay_60_3cm:.4f}")

# === BFI at flow=0 — should be zero but isn't ===
print("\n=== BFI at flow=(0,0) — should ideally be 0 ===")
print(f"  BFI 1cm: mean={np.mean(ds[4,zero_mask]):.4f}, std={np.std(ds[4,zero_mask]):.4f}")
print(f"  BFI 2cm: mean={np.mean(ds[5,zero_mask]):.4f}, std={np.std(ds[5,zero_mask]):.4f}")
print(f"  BFI 3cm: mean={np.mean(ds[6,zero_mask]):.4f}, std={np.std(ds[6,zero_mask]):.4f}")

# === Measured flow at pump=0 ===
print("\n=== Measured outlet flow at preset=(0,0) ===")
print(f"  Shallow: mean={np.mean(ds[2,zero_mask]):.4f}, std={np.std(ds[2,zero_mask]):.4f}")
print(f"  Deep:    mean={np.mean(ds[3,zero_mask]):.4f}, std={np.std(ds[3,zero_mask]):.4f}")

# === Variability within flow=0 trials ===
print("\n=== g2 variability within (0,0) trials ===")
g2_zero_1cm = g2[0, :, zero_mask]  # (501, 210)
print(f"  SDS 1cm: g2 std across trials at tau=0: {np.std(g2_zero_1cm[0,:]):.4f}")
print(f"  SDS 1cm: g2 std across trials at tau=250us: {np.std(g2_zero_1cm[100,:]):.4f}")
print(f"  SDS 1cm: g2 std across trials at tau=500us: {np.std(g2_zero_1cm[-1,:]):.4f}")
print(f"  SDS 1cm: g2 range at tau=0: {g2_zero_1cm[0,:].min():.4f} - {g2_zero_1cm[0,:].max():.4f}")
print(f"  SDS 1cm: g2 range at tau=500us: {g2_zero_1cm[-1,:].min():.4f} - {g2_zero_1cm[-1,:].max():.4f}")
