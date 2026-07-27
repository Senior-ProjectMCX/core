import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import scipy.io


# ==================================
# Project Paths
# ==================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = PROJECT_ROOT / "code"

# เพิ่มโฟลเดอร์ code เข้า Python path
# เพื่อไม่ต้องแก้โค้ดของเพื่อน
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))


from generate_dataset import generate_synthetic_dataset


# ==================================
# Capture Terminal Progress
# ==================================

class ProgressOutput:

    def __init__(self, progress_callback=None):

        self.progress_callback = progress_callback
        self.buffer = ""

    def write(self, text):

        # แสดงข้อความใน Terminal ตามเดิม
        sys.__stdout__.write(text)
        sys.__stdout__.flush()

        self.buffer += text

        while "\n" in self.buffer:

            line, self.buffer = self.buffer.split(
                "\n",
                1
            )

            self.process_line(
                line.strip()
            )

    def flush(self):

        sys.__stdout__.flush()

    def process_line(self, line):

        if not line:
            return

        if self.progress_callback is None:
            return

        if line.startswith(
            "Loading physical dataset"
        ):

            self.progress_callback(
                "Loading physical dataset...",
                3
            )

        elif line.startswith(
            "Loading simulated pathlengths"
        ):

            self.progress_callback(
                "Loading simulated pathlengths...",
                6
            )

        elif line.startswith(
            "Fitted coherence factors"
        ):

            self.progress_callback(
                "Calculating coherence factors...",
                9
            )

        elif line.startswith(
            "Generating synthetic g2 curves"
        ):

            match = re.search(
                r"for\s+(\d+)\s+trials",
                line
            )

            if match:

                total = int(
                    match.group(1)
                )

                self.progress_callback(
                    f"Generating g2 curves: 0/{total}",
                    10
                )

        elif line.startswith("Processed"):

            match = re.search(
                r"Processed\s+(\d+)/(\d+)",
                line
            )

            if match:

                current = int(
                    match.group(1)
                )

                total = int(
                    match.group(2)
                )

                # ช่วงคำนวณใช้ 10% ถึง 95%
                percentage = (
                    10
                    + (current / total) * 85
                )

                percentage = min(
                    float(percentage),
                    95
                )

                self.progress_callback(
                    (
                        f"Generating g2 curves: "
                        f"{current}/{total}"
                    ),
                    percentage
                )

        elif line.startswith(
            "Synthetic dataset saved"
        ):

            self.progress_callback(
                "Saving generated dataset...",
                97
            )


class Pipeline:

    def __init__(self):

        self.project_root = PROJECT_ROOT

        self.pathlengths_path = (
            self.project_root
            / "data"
            / "pathlengths.npz"
        )

        self.physical_mat_path = (
            self.project_root
            / "data"
            / "Dataset_phantom-intra40.mat"
        )

        self.output_mat_path = (
            self.project_root
            / "data"
            / "Dataset_simulated-intra40.mat"
        )

        self.dataset_set = None
        self.g2_data = None
        self.tau_array = None

    # ==================================
    # Generate and Load Dataset
    # ==================================

    def generate_dataset(
        self,
        progress_callback=None
    ):

        if not self.pathlengths_path.exists():

            raise FileNotFoundError(
                "Pathlength file not found:\n"
                f"{self.pathlengths_path}"
            )

        if not self.physical_mat_path.exists():

            raise FileNotFoundError(
                "Physical dataset not found:\n"
                f"{self.physical_mat_path}"
            )

        self.output_mat_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if progress_callback is not None:

            progress_callback(
                "Preparing dataset generation...",
                0
            )

        progress_output = ProgressOutput(
            progress_callback
        )

        # ดักข้อความ print ของโค้ดเดิม
        # โดยไม่ต้องแก้ไฟล์ของเพื่อน
        with redirect_stdout(progress_output):

            generate_synthetic_dataset(
                pathlengths_path=str(
                    self.pathlengths_path
                ),
                mat_path=str(
                    self.physical_mat_path
                ),
                output_mat_path=str(
                    self.output_mat_path
                )
            )

        if progress_callback is not None:

            progress_callback(
                "Loading generated dataset...",
                98
            )

        self.load_dataset()

        if progress_callback is not None:

            progress_callback(
                "Dataset ready.",
                100
            )

        return self.get_dataset_size()

    # ==================================
    # Load Existing Dataset
    # ==================================

    def load_dataset(self):

        if not self.output_mat_path.exists():

            raise FileNotFoundError(
                "Generated dataset not found:\n"
                f"{self.output_mat_path}"
            )

        mat_data = scipy.io.loadmat(
            self.output_mat_path
        )

        if "Dataset_set_intra40" not in mat_data:

            raise KeyError(
                "Dataset_set_intra40 "
                "was not found in the MAT file."
            )

        if "ave_gtau_intra40" not in mat_data:

            raise KeyError(
                "ave_gtau_intra40 "
                "was not found in the MAT file."
            )

        self.dataset_set = np.asarray(
            mat_data["Dataset_set_intra40"]
        )

        self.g2_data = np.asarray(
            mat_data["ave_gtau_intra40"]
        )

        number_of_tau_points = (
            self.g2_data.shape[1]
        )

        self.tau_array = np.linspace(
            0,
            500,
            number_of_tau_points
        )

        return self.get_dataset_size()

    # ==================================
    # Dataset Status
    # ==================================

    def is_dataset_loaded(self):

        return (
            self.dataset_set is not None
            and self.g2_data is not None
        )

    def get_dataset_size(self):

        if not self.is_dataset_loaded():
            return 0

        return int(
            self.dataset_set.shape[1]
        )

    # ==================================
    # Flow Values
    # ==================================

    def get_all_shallow_values(self):

        if not self.is_dataset_loaded():
            return []

        values = np.unique(
            self.dataset_set[0, :]
        )

        return [
            float(value)
            for value in values
        ]

    def get_all_deep_values(self):

        if not self.is_dataset_loaded():
            return []

        values = np.unique(
            self.dataset_set[1, :]
        )

        return [
            float(value)
            for value in values
        ]

    def get_deep_values_for_shallow(
        self,
        shallow_flow
    ):

        if not self.is_dataset_loaded():
            return []

        shallow_flow = float(
            shallow_flow
        )

        matching_mask = np.isclose(
            self.dataset_set[0, :],
            shallow_flow,
            rtol=0,
            atol=1e-6
        )

        values = np.unique(
            self.dataset_set[
                1,
                matching_mask
            ]
        )

        return [
            float(value)
            for value in values
        ]

    def get_shallow_values_for_deep(
        self,
        deep_flow
    ):

        if not self.is_dataset_loaded():
            return []

        deep_flow = float(
            deep_flow
        )

        matching_mask = np.isclose(
            self.dataset_set[1, :],
            deep_flow,
            rtol=0,
            atol=1e-6
        )

        values = np.unique(
            self.dataset_set[
                0,
                matching_mask
            ]
        )

        return [
            float(value)
            for value in values
        ]

    # ==================================
    # Matching Trials
    # ==================================

    def get_matching_trials(
        self,
        shallow_flow,
        deep_flow
    ):

        if not self.is_dataset_loaded():
            return []

        shallow_flow = float(
            shallow_flow
        )

        deep_flow = float(
            deep_flow
        )

        shallow_matches = np.isclose(
            self.dataset_set[0, :],
            shallow_flow,
            rtol=0,
            atol=1e-6
        )

        deep_matches = np.isclose(
            self.dataset_set[1, :],
            deep_flow,
            rtol=0,
            atol=1e-6
        )

        matching_indices = np.where(
            shallow_matches
            & deep_matches
        )[0]

        return [
            int(index) + 1
            for index in matching_indices
        ]

    # ==================================
    # Dataset Rows
    # ==================================

    def get_all_dataset_rows(self):

        if not self.is_dataset_loaded():
            return []

        rows = []

        number_of_trials = (
            self.get_dataset_size()
        )

        for index in range(
            number_of_trials
        ):

            rows.append({
                "index": index,
                "trial": index + 1,

                "preset_shallow_flow": float(
                    self.dataset_set[0, index]
                ),

                "preset_deep_flow": float(
                    self.dataset_set[1, index]
                ),

                "measured_shallow_flow": float(
                    self.dataset_set[2, index]
                ),

                "measured_deep_flow": float(
                    self.dataset_set[3, index]
                ),

                "physical_bfi_1cm": float(
                    self.dataset_set[4, index]
                ),

                "physical_bfi_2cm": float(
                    self.dataset_set[5, index]
                ),

                "physical_bfi_3cm": float(
                    self.dataset_set[6, index]
                )
            })

        return rows

    # ==================================
    # Get Trial Result
    # ==================================

    def get_trial_result(self, trial):

        if not self.is_dataset_loaded():

            raise RuntimeError(
                "Dataset has not been loaded."
            )

        trial = int(trial)
        index = trial - 1

        if (
            index < 0
            or index >= self.get_dataset_size()
        ):

            raise IndexError(
                f"Trial {trial} is outside "
                "the dataset range."
            )

        return {
            "index": index,
            "trial": trial,

            "preset_shallow_flow": float(
                self.dataset_set[0, index]
            ),

            "preset_deep_flow": float(
                self.dataset_set[1, index]
            ),

            "measured_shallow_flow": float(
                self.dataset_set[2, index]
            ),

            "measured_deep_flow": float(
                self.dataset_set[3, index]
            ),

            "physical_bfi_1cm": float(
                self.dataset_set[4, index]
            ),

            "physical_bfi_2cm": float(
                self.dataset_set[5, index]
            ),

            "physical_bfi_3cm": float(
                self.dataset_set[6, index]
            ),

            "tau": self.tau_array,

            "g2_1cm": self.g2_data[
                0,
                :,
                index
            ],

            "g2_2cm": self.g2_data[
                1,
                :,
                index
            ],

            "g2_3cm": self.g2_data[
                2,
                :,
                index
            ]
        }