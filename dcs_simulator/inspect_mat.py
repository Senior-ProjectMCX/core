from pathlib import Path
from scipy.io import loadmat

file_path = (
    Path(__file__).resolve().parent
    / "reference_data"
    / "Dataset_phantom-intra40.mat"
)

data = loadmat(file_path)

print("=== MAT FILE METADATA ===")

for key, value in data.items():
    print(f"\nKEY: {key}")

    if key.startswith("__"):
        print(value)
        continue

    print("Type:", type(value))
    print("Shape:", getattr(value, "shape", None))
    print("Dtype:", getattr(value, "dtype", None))