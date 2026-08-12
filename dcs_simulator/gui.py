"""
DCS Simulator GUI

Tab 1: Reference Validation
    - Select only flow conditions available in the physical reference dataset
    - Generate model g2
    - Compare against physical reference
    - Display MAE, RMSE and maximum absolute error

Tab 2: Human Dataset Generator
    - Generate subject-specific synthetic DCS g2 datasets
    - Inputs: sex, measured thigh SAT, optional measured skin thickness
    - Flow grid is generated internally

NOTE:
The current flow -> alphaDb mapping is temporary and will be
replaced after final calibration.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import threading


import numpy as np
from scipy.io import loadmat, savemat

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from reference_loader import (
    load_reference_condition,
    REFERENCE_FILE,
)

from dcs_processor import (
    calculate_g2,
    extract_photon_data,
    K0,
)

from parameters import MU_A_MM

from baseline_model import (
    calculate_baseline_g2,
    load_baseline_pathlengths,
)

from human_model import HumanInput
from dataset_generator import generate_human_dataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DETP_FILE = (
    BASE_DIR
    / "output"
    / "detp_core_v1.npy"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

FLOW_TO_ALPHA_DB_SCALE = 1e-6

TAU_US = np.arange(
    0,
    501,
    dtype=float,
)

TAU_SECONDS = (
    TAU_US
    * 1e-6
)


# ============================================================
# HELPERS
# ============================================================

def format_flow_value(value):
    """
    Display flow values cleanly.

    Example:
        5.0  -> "5"
        2.5  -> "2.5"
    """

    value = float(value)

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


# ============================================================
# LOAD AVAILABLE REFERENCE CONDITIONS
# ============================================================

def load_reference_conditions():
    """
    Read all unique commanded shallow/deep flow combinations
    available in the physical reference MAT file.

    Dataset_set_intra40:
        row 0 = commanded shallow flow
        row 1 = commanded deep flow
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference dataset not found:\n"
            f"{REFERENCE_FILE}"
        )

    data = loadmat(
        REFERENCE_FILE
    )

    if "Dataset_set_intra40" not in data:
        raise KeyError(
            "Dataset_set_intra40 not found "
            "in reference MAT file."
        )

    dataset = data[
        "Dataset_set_intra40"
    ]

    if (
        dataset.ndim != 2
        or dataset.shape[0] < 2
    ):
        raise ValueError(
            "Unexpected Dataset_set_intra40 shape: "
            f"{dataset.shape}"
        )

    shallow_values = dataset[0]
    deep_values = dataset[1]

    conditions = {}

    for shallow, deep in zip(
        shallow_values,
        deep_values,
    ):

        shallow = float(shallow)
        deep = float(deep)

        if shallow not in conditions:
            conditions[shallow] = set()

        conditions[shallow].add(
            deep
        )

    # Sort shallow and deep values
    sorted_conditions = {}

    for shallow in sorted(
        conditions.keys()
    ):

        sorted_conditions[shallow] = sorted(
            conditions[shallow]
        )

    return sorted_conditions


# ============================================================
# FLOW -> alphaDb
# ============================================================

def flow_to_alpha_db(flow):
    """
    Temporary Core V1 mapping.

        alphaDb = flow * 1e-6 mm^2/s

    This function will be replaced after final calibration.
    """

    flow = float(flow)

    if not np.isfinite(flow):
        raise ValueError(
            "Flow must be finite."
        )

    if flow < 0:
        raise ValueError(
            "Flow must be >= 0."
        )

    return (
        flow
        * FLOW_TO_ALPHA_DB_SCALE
    )


# ============================================================
# DYNAMIC g1
# ============================================================

def calculate_dynamic_g1(
    detp,
    tau_seconds,
    detector_id,
    alpha_db_shallow,
    alpha_db_deep,
):
    """
    Calculate absorption-weighted normalized g1(tau)
    for one detector.
    """

    photon_data = extract_photon_data(
        detp
    )

    detector_ids = (
        photon_data[
            "detector_id"
        ]
    )

    mask = (
        detector_ids
        == detector_id
    )

    if not np.any(mask):
        raise ValueError(
            f"No photons detected at detector "
            f"{detector_id}"
        )

    path_shallow = (
        photon_data[
            "path_shallow"
        ][mask]
    )

    path_deep = (
        photon_data[
            "path_deep"
        ][mask]
    )

    momentum_shallow = (
        photon_data[
            "momentum_shallow"
        ][mask]
    )

    momentum_deep = (
        photon_data[
            "momentum_deep"
        ][mask]
    )

    # --------------------------------------------------------
    # Absorption weighting
    # --------------------------------------------------------

    absorption_weight = np.exp(
        -MU_A_MM
        * (
            path_shallow
            + path_deep
        )
    )

    weight_sum = np.sum(
        absorption_weight
    )

    if (
        not np.isfinite(weight_sum)
        or weight_sum <= 0
    ):
        raise ValueError(
            "Invalid absorption weights."
        )

    # --------------------------------------------------------
    # Dynamic decay
    # --------------------------------------------------------

    decay_rate = (
        2.0
        * K0**2
        * (
            momentum_shallow
            * alpha_db_shallow
            +
            momentum_deep
            * alpha_db_deep
        )
    )

    g1 = np.empty(
        len(tau_seconds),
        dtype=float,
    )

    for index, tau in enumerate(
        tau_seconds
    ):

        dynamic_term = np.exp(
            -decay_rate
            * tau
        )

        g1[index] = (
            np.sum(
                absorption_weight
                * dynamic_term
            )
            / weight_sum
        )

    g1 = (
        g1
        / g1[0]
    )

    if not np.all(
        np.isfinite(g1)
    ):
        raise ValueError(
            "Generated g1 contains "
            "non-finite values."
        )

    return g1


# ============================================================
# RUN VALIDATION
# ============================================================

def run_validation(
    shallow_flow,
    deep_flow,
):
    """
    Compare:
      1) Proposed model (this project)
      2) Baseline pathlength model
      3) Physical reference
    for one known reference condition.
    """

    shallow_flow = float(shallow_flow)
    deep_flow = float(deep_flow)

    alpha_db_shallow = flow_to_alpha_db(shallow_flow)
    alpha_db_deep = flow_to_alpha_db(deep_flow)

    if not DETP_FILE.exists():
        raise FileNotFoundError(
            "Detected-photon file not found:\n"
            f"{DETP_FILE}\n\n"
            "Run simulation.py first."
        )

    detp = np.load(DETP_FILE)

    if detp.ndim != 2 or detp.shape[0] != 5:
        raise ValueError(
            "Expected detp shape (5, N). "
            f"Got {detp.shape}"
        )

    reference = load_reference_condition(
        shallow_flow=shallow_flow,
        deep_flow=deep_flow,
    )

    baseline_paths = load_baseline_pathlengths()
    results = {}
    detector_ids = detp[0].astype(int)

    all_proposed_errors = []
    all_baseline_errors = []

    for sds_cm in (1, 2, 3):
        reference_g2 = reference["g2"][sds_cm]["mean"]
        beta = float(reference_g2[0] - 1.0)

        # Proposed model
        g1 = calculate_dynamic_g1(
            detp=detp,
            tau_seconds=TAU_SECONDS,
            detector_id=sds_cm,
            alpha_db_shallow=alpha_db_shallow,
            alpha_db_deep=alpha_db_deep,
        )

        generated_g2 = calculate_g2(
            g1=g1,
            beta=beta,
        )

        # Baseline model from the comparison code
        Ls, Ld = baseline_paths[sds_cm]

        baseline_g2 = calculate_baseline_g2(
            Ls=Ls,
            Ld=Ld,
            flow_s=shallow_flow,
            flow_d=deep_flow,
            tau_array=TAU_SECONDS,
            beta=beta,
        )

        proposed_error = generated_g2 - reference_g2
        baseline_error = baseline_g2 - reference_g2

        proposed_mae = float(np.mean(np.abs(proposed_error)))
        proposed_rmse = float(np.sqrt(np.mean(proposed_error ** 2)))
        proposed_max_error = float(np.max(np.abs(proposed_error)))

        baseline_mae = float(np.mean(np.abs(baseline_error)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
        baseline_max_error = float(np.max(np.abs(baseline_error)))

        all_proposed_errors.append(proposed_error)
        all_baseline_errors.append(baseline_error)

        if np.isclose(proposed_rmse, baseline_rmse, rtol=1e-6, atol=1e-12):
            better = "Tie"
        elif proposed_rmse < baseline_rmse:
            better = "Proposed"
        else:
            better = "Baseline"

        photon_count = int(np.sum(detector_ids == sds_cm))

        results[sds_cm] = {
            "beta": beta,
            "g1": g1,
            "generated_g2": generated_g2,
            "baseline_g2": baseline_g2,
            "reference_g2": reference_g2,
            "mae": proposed_mae,
            "rmse": proposed_rmse,
            "max_error": proposed_max_error,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_max_error": baseline_max_error,
            "better": better,
            "photons": photon_count,
        }

    proposed_all = np.concatenate(all_proposed_errors)
    baseline_all = np.concatenate(all_baseline_errors)

    proposed_combined_rmse = float(
        np.sqrt(np.mean(proposed_all ** 2))
    )
    baseline_combined_rmse = float(
        np.sqrt(np.mean(baseline_all ** 2))
    )

    if np.isclose(
        proposed_combined_rmse,
        baseline_combined_rmse,
        rtol=1e-6,
        atol=1e-12,
    ):
        overall_better = "Tie"
        verdict = (
            "Result: Proposed and Baseline models have "
            "approximately equal combined RMSE for this condition."
        )
    elif proposed_combined_rmse < baseline_combined_rmse:
        overall_better = "Proposed"
        improvement = (
            (baseline_combined_rmse - proposed_combined_rmse)
            / baseline_combined_rmse
            * 100.0
        )
        verdict = (
            "Result: Proposed model has lower combined RMSE "
            f"for this condition ({improvement:.2f}% lower than Baseline)."
        )
    else:
        overall_better = "Baseline"
        improvement = (
            (proposed_combined_rmse - baseline_combined_rmse)
            / proposed_combined_rmse
            * 100.0
        )
        verdict = (
            "Result: Baseline model has lower combined RMSE "
            f"for this condition ({improvement:.2f}% lower than Proposed)."
        )

    return {
        "detp": detp,
        "reference": reference,
        "results": results,
        "shallow_flow": shallow_flow,
        "deep_flow": deep_flow,
        "alpha_db_shallow": alpha_db_shallow,
        "alpha_db_deep": alpha_db_deep,
        "proposed_combined_rmse": proposed_combined_rmse,
        "baseline_combined_rmse": baseline_combined_rmse,
        "overall_better": overall_better,
        "verdict": verdict,
    }


# ============================================================
# GUI
# ============================================================

class DCSSimulatorGUI:

    def __init__(
        self,
        root,
    ):

        self.root = root

        self.root.title(
            "DCS Simulator"
        )

        self.root.geometry(
            "1450x850"
        )

        self.root.minsize(
            1150,
            720
        )

        self.last_validation_output = None
        self.last_human_output = None
        self.human_dataset = None

        # ----------------------------------------------------
        # Load reference conditions
        # ----------------------------------------------------

        try:

            self.reference_conditions = (
                load_reference_conditions()
            )

        except Exception as error:

            messagebox.showerror(
                "Reference Dataset Error",
                str(error),
            )

            self.reference_conditions = {}

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = ttk.Frame(
            root,
            padding=(
                18,
                14,
                18,
                8,
            ),
        )

        header.pack(
            fill="x"
        )

        ttk.Label(
            header,
            text="DCS Simulator",
            font=(
                "Segoe UI",
                22,
                "bold",
            ),
        ).pack(
            side="left"
        )

        ttk.Label(
            header,
            text=(
                "Validation & Synthetic Dataset Generation"
            ),
            font=(
                "Segoe UI",
                10,
            ),
        ).pack(
            side="left",
            padx=(15, 0),
            pady=(8, 0),
        )

        # ----------------------------------------------------
        # Notebook
        # ----------------------------------------------------

        self.notebook = ttk.Notebook(
            root
        )

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15),
        )

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------

        self.validation_tab = ttk.Frame(
            self.notebook,
            padding=15,
        )

        self.human_tab = ttk.Frame(
            self.notebook,
            padding=15,
        )

        self.notebook.add(
            self.validation_tab,
            text="  Model Validation & Benchmark  ",
        )

        self.notebook.add(
            self.human_tab,
            text="  Human Dataset Generator  ",
        )

        # ----------------------------------------------------
        # Build
        # ----------------------------------------------------

        self.build_validation_tab()
        self.build_human_tab()


    # ========================================================
    # TAB 1
    # ========================================================

    def build_validation_tab(
        self,
    ):

        self.validation_tab.columnconfigure(
            0,
            weight=0,
        )

        self.validation_tab.columnconfigure(
            1,
            weight=1,
        )

        self.validation_tab.rowconfigure(
            0,
            weight=1,
        )

        # ----------------------------------------------------
        # Left panel
        # ----------------------------------------------------

        left = ttk.Frame(
            self.validation_tab,
            padding=10,
        )

        left.grid(
            row=0,
            column=0,
            sticky="ns",
            padx=(0, 15),
        )

        ttk.Label(
            left,
            text="Model Validation & Benchmark",
            font=(
                "Segoe UI",
                18,
                "bold",
            ),
        ).pack(
            anchor="w"
        )

        ttk.Label(
            left,
            text=(
                "Compare the DCS model against "
                "physical measurements."
            ),
            wraplength=260,
        ).pack(
            anchor="w",
            pady=(5, 20),
        )

        # ----------------------------------------------------
        # Condition box
        # ----------------------------------------------------

        condition_frame = (
            ttk.LabelFrame(
                left,
                text="Reference Condition",
                padding=15,
            )
        )

        condition_frame.pack(
            fill="x"
        )

        condition_frame.columnconfigure(
            0,
            weight=1,
        )

        # ----------------------------------------------------
        # Shallow dropdown
        # ----------------------------------------------------

        ttk.Label(
            condition_frame,
            text="Shallow Flow",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.validation_shallow_var = (
            tk.StringVar()
        )

        shallow_values = [
            format_flow_value(value)
            for value
            in self.reference_conditions.keys()
        ]

        self.validation_shallow_combo = (
            ttk.Combobox(
                condition_frame,
                textvariable=(
                    self.validation_shallow_var
                ),
                values=shallow_values,
                state="readonly",
                width=18,
            )
        )

        self.validation_shallow_combo.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 3),
        )

        self.validation_shallow_combo.set(
            "Select shallow flow"
        )

        self.validation_shallow_combo.bind(
            "<<ComboboxSelected>>",
            self.on_shallow_changed,
        )

        ttk.Label(
            condition_frame,
            text="ml/min/100g",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(0, 14),
        )

        # ----------------------------------------------------
        # Deep dropdown
        # ----------------------------------------------------

        ttk.Label(
            condition_frame,
            text="Deep Flow",
        ).grid(
            row=3,
            column=0,
            sticky="w",
        )

        self.validation_deep_var = (
            tk.StringVar()
        )

        self.validation_deep_combo = (
            ttk.Combobox(
                condition_frame,
                textvariable=(
                    self.validation_deep_var
                ),
                state="readonly",
                width=18,
            )
        )

        self.validation_deep_combo.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(4, 3),
        )

        self.validation_deep_combo.set(
            "Select shallow first"
        )

        ttk.Label(
            condition_frame,
            text="ml/min/100g",
        ).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(0, 15),
        )

        self.validation_run_button = (
            ttk.Button(
                condition_frame,
                text="Generate & Compare",
                command=(
                    self.run_reference_validation
                ),
            )
        )

        self.validation_run_button.grid(
            row=6,
            column=0,
            sticky="ew",
        )

        # ----------------------------------------------------
        # Status box
        # ----------------------------------------------------

        status_frame = ttk.LabelFrame(
            left,
            text="Validation Status",
            padding=15,
        )

        status_frame.pack(
            fill="x",
            pady=(18, 0),
        )

        self.validation_status_var = (
            tk.StringVar(
                value="Ready"
            )
        )

        ttk.Label(
            status_frame,
            textvariable=(
                self.validation_status_var
            ),
            wraplength=250,
        ).pack(
            anchor="w"
        )

        self.validation_condition_var = (
            tk.StringVar(
                value="Condition: -"
            )
        )

        ttk.Label(
            status_frame,
            textvariable=(
                self.validation_condition_var
            ),
        ).pack(
            anchor="w",
            pady=(10, 0),
        )

        self.validation_alpha_var = (
            tk.StringVar(
                value="alphaDb: -"
            )
        )

        ttk.Label(
            status_frame,
            textvariable=(
                self.validation_alpha_var
            ),
            wraplength=250,
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

        self.validation_photon_var = (
            tk.StringVar(
                value="Detected photons: -"
            )
        )

        ttk.Label(
            status_frame,
            textvariable=(
                self.validation_photon_var
            ),
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

        # ----------------------------------------------------
        # Right panel
        # ----------------------------------------------------

        right = ttk.Frame(
            self.validation_tab,
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        right.columnconfigure(
            0,
            weight=1,
        )

        right.rowconfigure(
            1,
            weight=1,
        )

        graph_header = ttk.Frame(right)
        graph_header.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )
        graph_header.columnconfigure(0, weight=1)

        ttk.Label(
            graph_header,
            text="Proposed vs Baseline vs Physical Reference",
            font=("Segoe UI", 15, "bold"),
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Label(
            graph_header,
            text="Graph View:",
        ).grid(
            row=0,
            column=1,
            padx=(10, 6),
        )

        self.sds_view_var = tk.StringVar(value="1 cm")

        self.sds_view_combo = ttk.Combobox(
            graph_header,
            textvariable=self.sds_view_var,
            values=["1 cm", "2 cm", "3 cm", "All"],
            state="readonly",
            width=8,
        )
        self.sds_view_combo.grid(
            row=0,
            column=2,
            sticky="e",
        )
        self.sds_view_combo.bind(
            "<<ComboboxSelected>>",
            self.on_sds_view_changed,
        )

        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        self.validation_figure = Figure(
            figsize=(9, 5),
            dpi=100,
        )

        self.validation_axis = (
            self.validation_figure.add_subplot(
                111
            )
        )

        self.validation_axis.set_xlabel(
            "Delay time τ (μs)"
        )

        self.validation_axis.set_ylabel(
            "g₂(τ)"
        )

        self.validation_axis.set_title(
            "Select a reference condition"
        )

        self.validation_axis.grid(
            alpha=0.25
        )

        self.validation_canvas = (
            FigureCanvasTkAgg(
                self.validation_figure,
                master=right,
            )
        )

        self.validation_canvas.get_tk_widget().grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics_frame = ttk.LabelFrame(
            right,
            text="Validation Metrics",
            padding=10,
        )

        metrics_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

        columns = (
            "sds",
            "proposed_rmse",
            "baseline_rmse",
            "better",
        )

        self.validation_metrics_table = (
            ttk.Treeview(
                metrics_frame,
                columns=columns,
                show="headings",
                height=4,
            )
        )

        headings = {
            "sds": "SDS",
            "proposed_rmse": "Proposed RMSE",
            "baseline_rmse": "Baseline RMSE",
            "better": "Better Match",
        }

        for column in columns:

            self.validation_metrics_table.heading(
                column,
                text=headings[column],
            )

            self.validation_metrics_table.column(
                column,
                width=120,
                anchor="center",
            )

        self.validation_metrics_table.pack(
            fill="x"
        )

        self.validation_verdict_var = tk.StringVar(
            value="Run a condition to compare Proposed and Baseline models."
        )

        ttk.Label(
            metrics_frame,
            textvariable=self.validation_verdict_var,
            wraplength=900,
            font=("Segoe UI", 10, "bold"),
        ).pack(
            anchor="w",
            pady=(8, 0),
        )


    # ========================================================
    # REFERENCE DROPDOWNS
    # ========================================================

    def on_shallow_changed(
        self,
        event=None,
    ):

        self.update_deep_dropdown()


    def update_deep_dropdown(
        self,
    ):

        shallow_text = (
            self.validation_shallow_var.get()
        )

        if not shallow_text:
            return

        shallow = float(
            shallow_text
        )

        # Find exact numeric key
        matched_key = None

        for key in (
            self.reference_conditions.keys()
        ):

            if np.isclose(
                key,
                shallow,
            ):
                matched_key = key
                break

        if matched_key is None:

            self.validation_deep_combo[
                "values"
            ] = []

            self.validation_deep_var.set(
                ""
            )

            return

        deep_values = [
            format_flow_value(value)
            for value
            in self.reference_conditions[
                matched_key
            ]
        ]

        self.validation_deep_combo[
            "values"
        ] = deep_values

        if deep_values:

            # Prefer 20 when available
            if "20" in deep_values:
                selected = "20"
            else:
                selected = deep_values[0]

            self.validation_deep_var.set(
                selected
            )

        else:

            self.validation_deep_var.set(
                ""
            )


    # ========================================================
    # RUN REFERENCE VALIDATION
    # ========================================================

    def run_reference_validation(
        self,
    ):

        try:

            shallow_text = (
                self.validation_shallow_var.get()
            )

            deep_text = (
                self.validation_deep_var.get()
            )

            if (
                not shallow_text
                or not deep_text
            ):
                raise ValueError(
                    "Select both shallow "
                    "and deep flow."
                )

            shallow_flow = float(
                shallow_text
            )

            deep_flow = float(
                deep_text
            )

            self.validation_run_button.config(
                state="disabled"
            )

            self.validation_status_var.set(
                "Processing..."
            )

            self.root.update_idletasks()

            output = run_validation(
                shallow_flow=shallow_flow,
                deep_flow=deep_flow,
            )

            results = (
                output["results"]
            )

            self.last_validation_output = output

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            self.validation_condition_var.set(
                "Condition: "
                f"{shallow_flow:g} / "
                f"{deep_flow:g}"
            )

            self.validation_alpha_var.set(
                "alphaDb: "
                f"{output['alpha_db_shallow']:.3e} / "
                f"{output['alpha_db_deep']:.3e} mm²/s"
            )

            self.validation_photon_var.set(
                "Detected photons: "
                f"{output['detp'].shape[1]:,}"
            )

            # ------------------------------------------------
            # Graph
            # ------------------------------------------------

            self.update_validation_graph(
                results,
                shallow_flow,
                deep_flow,
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            self.update_validation_metrics(
                results,
                output,
            )

            self.validation_status_var.set(
                "Complete"
            )

        except Exception as error:

            self.validation_status_var.set(
                "Error"
            )

            messagebox.showerror(
                "Validation Error",
                str(error),
            )

        finally:

            self.validation_run_button.config(
                state="normal"
            )


    # ========================================================
    # VALIDATION GRAPH
    # ========================================================

    def on_sds_view_changed(
        self,
        event=None,
    ):
        """
        Redraw the graph immediately using the latest validation
        result. No simulation rerun is required.
        """
        if self.last_validation_output is None:
            return

        output = self.last_validation_output

        self.update_validation_graph(
            output["results"],
            output["shallow_flow"],
            output["deep_flow"],
        )


    def update_validation_graph(
        self,
        results,
        shallow_flow,
        deep_flow,
    ):

        self.validation_axis.clear()

        view = self.sds_view_var.get()

        if view == "All":
            selected_sds = (1, 2, 3)
        else:
            selected_sds = (int(view.split()[0]),)

        for sds_cm in selected_sds:
            generated = results[sds_cm]["generated_g2"]
            baseline = results[sds_cm]["baseline_g2"]
            reference = results[sds_cm]["reference_g2"]

            self.validation_axis.plot(
                TAU_US,
                generated,
                linewidth=2.0,
                label=f"Proposed SDS {sds_cm} cm",
            )

            self.validation_axis.plot(
                TAU_US,
                baseline,
                linestyle=":",
                linewidth=2.0,
                label=f"Baseline SDS {sds_cm} cm",
            )

            self.validation_axis.plot(
                TAU_US,
                reference,
                linestyle="--",
                linewidth=2.0,
                label=f"Physical Reference SDS {sds_cm} cm",
            )

        self.validation_axis.set_xlabel(
            "Delay time τ (μs)"
        )

        self.validation_axis.set_ylabel(
            "g₂(τ)"
        )

        if view == "All":
            view_title = "All SDS"
        else:
            view_title = f"SDS {view}"

        self.validation_axis.set_title(
            f"{view_title}\\n"
            f"Shallow = {shallow_flow:g}, "
            f"Deep = {deep_flow:g}"
        )

        self.validation_axis.grid(alpha=0.25)

        self.validation_axis.legend(
            fontsize=8,
            ncol=1 if view != "All" else 3,
        )

        self.validation_figure.tight_layout()
        self.validation_canvas.draw()


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    def update_validation_metrics(
        self,
        results,
        output,
    ):

        for item in self.validation_metrics_table.get_children():
            self.validation_metrics_table.delete(item)

        for sds_cm in (1, 2, 3):
            data = results[sds_cm]

            self.validation_metrics_table.insert(
                "",
                "end",
                values=(
                    f"{sds_cm} cm",
                    f"{data['rmse']:.6f}",
                    f"{data['baseline_rmse']:.6f}",
                    (
                        f"{data['better']} ✓"
                        if data["better"] != "Tie"
                        else "Tie"
                    ),
                ),
            )

        self.validation_metrics_table.insert(
            "",
            "end",
            values=(
                "Combined",
                f"{output['proposed_combined_rmse']:.6f}",
                f"{output['baseline_combined_rmse']:.6f}",
                (
                    f"{output['overall_better']} ✓"
                    if output["overall_better"] != "Tie"
                    else "Tie"
                ),
            ),
        )

        proposed = output["proposed_combined_rmse"]
        baseline = output["baseline_combined_rmse"]

        if output["overall_better"] == "Proposed":
            percent = (
                (baseline - proposed)
                / baseline
                * 100.0
            )
            verdict = (
                "✓ Overall: Proposed model is the better match for this condition.  "
                f"Combined RMSE: {proposed:.6f} vs {baseline:.6f}.  "
                f"Proposed RMSE is {percent:.2f}% lower than Baseline."
            )
        elif output["overall_better"] == "Baseline":
            percent = (
                (proposed - baseline)
                / proposed
                * 100.0
            )
            verdict = (
                "✓ Overall: Baseline model is the better match for this condition.  "
                f"Combined RMSE: {baseline:.6f} vs {proposed:.6f}.  "
                f"Baseline RMSE is {percent:.2f}% lower than Proposed."
            )
        else:
            verdict = (
                "Overall: Both models have approximately equal combined RMSE "
                "for this condition."
            )

        self.validation_verdict_var.set(verdict)


    # ========================================================
    # TAB 2
    # ========================================================

    def build_human_tab(self):
        self.human_tab.columnconfigure(0, weight=0)
        self.human_tab.columnconfigure(1, weight=1)
        self.human_tab.rowconfigure(0, weight=1)

        left = ttk.Frame(self.human_tab, padding=(18, 10, 18, 10))
        left.grid(row=0, column=0, sticky="ns", padx=(0, 15))

        ttk.Label(left, text="Human Dataset Generator",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(left,
                  text="Generate subject-specific synthetic DCS g₂ datasets.",
                  wraplength=290).pack(anchor="w", pady=(5, 18))

        subject = ttk.LabelFrame(left, text="Subject Parameters", padding=15)
        subject.pack(fill="x")
        subject.columnconfigure(0, weight=1)

        self.sex_var = tk.StringVar()
        self.sat_var = tk.StringVar()
        self.skin_var = tk.StringVar()

        self.human_placeholder_entries = {}

        def add_placeholder_entry(row, label, variable, placeholder, unit):
            ttk.Label(subject, text=label).grid(
                row=row, column=0, sticky="w"
            )

            entry = ttk.Entry(
                subject,
                textvariable=variable,
            )
            entry.grid(
                row=row + 1,
                column=0,
                sticky="ew",
                pady=(3, 2),
            )

            ttk.Label(subject, text=unit).grid(
                row=row + 2,
                column=0,
                sticky="w",
                pady=(0, 9),
            )

            self.human_placeholder_entries[entry] = {
                "variable": variable,
                "placeholder": placeholder,
                "active": False,
            }

            self._set_human_placeholder(entry)

            entry.bind(
                "<FocusIn>",
                lambda event, widget=entry:
                    self._clear_human_placeholder(widget),
            )
            entry.bind(
                "<FocusOut>",
                lambda event, widget=entry:
                    self._restore_human_placeholder(widget),
            )

        ttk.Label(subject, text="Sex *").grid(
            row=0, column=0, sticky="w"
        )

        self.sex_combo = ttk.Combobox(
            subject,
            textvariable=self.sex_var,
            values=["Female", "Male"],
            state="readonly",
        )
        self.sex_combo.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(3, 9),
        )
        self.sex_combo.set("Select sex")

        add_placeholder_entry(
            2,
            "Thigh SAT Thickness *",
            self.sat_var,
            "e.g. 10.0",
            "mm  •  measured at/near the DCS site",
        )

        add_placeholder_entry(
            5,
            "Skin Thickness (optional)",
            self.skin_var,
            "e.g. 1.7",
            "mm  •  leave blank to use sex-based reference",
        )

        action = ttk.LabelFrame(left, text="Dataset Generation", padding=15)
        action.pack(fill="x", pady=(15, 0))
        action.columnconfigure(0, weight=1)

        ttk.Label(
            action,
            text=(
                "Enter basic subject information. The simulator builds the "
                "human model and automatically generates a reference flow grid."
            ),
            wraplength=270,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.human_generate_button = ttk.Button(
            action,
            text="Generate Dataset",
            command=self.run_human_dataset,
        )
        self.human_generate_button.grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )

        self.human_export_csv_button = ttk.Button(
            action,
            text="Download CSV",
            command=self.export_human_dataset_csv,
            state="disabled",
        )
        self.human_export_csv_button.grid(
            row=2, column=0, sticky="ew", pady=(8, 0), padx=(0, 4)
        )

        self.human_export_mat_button = ttk.Button(
            action,
            text="Download MAT",
            command=self.export_human_dataset_mat,
            state="disabled",
        )
        self.human_export_mat_button.grid(
            row=2, column=1, sticky="ew", pady=(8, 0), padx=(4, 0)
        )

        status = ttk.LabelFrame(left, text="Generation Status", padding=15)
        status.pack(fill="x", pady=(15, 0))
        self.human_status_var = tk.StringVar(value="Ready")
        self.human_subject_var = tk.StringVar(value="Subject: -")
        self.human_geometry_var = tk.StringVar(value="Geometry: -")
        self.human_dataset_var = tk.StringVar(value="Dataset: -")
        for var in (
            self.human_status_var,
            self.human_subject_var,
            self.human_geometry_var,
            self.human_dataset_var,
        ):
            ttk.Label(
                status,
                textvariable=var,
                wraplength=285,
                justify="left",
            ).pack(
                anchor="w",
                fill="x",
                pady=(0, 3),
            )

        right = ttk.Frame(self.human_tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=4, minsize=250)
        right.rowconfigure(3, weight=6, minsize=330)

        # ====================================================
        # DATASET BROWSER
        # ====================================================

        browser_header = ttk.Frame(right)
        browser_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        browser_header.columnconfigure(0, weight=1)

        ttk.Label(
            browser_header,
            text="Dataset Browser",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            browser_header,
            text="Select a row to preview its flow condition.",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        table_frame = ttk.Frame(right)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = (
            "condition",
            "shallow",
            "deep",
            "sds",
            "tau",
            "g2",
        )

        self.human_dataset_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        headings = {
            "condition": "Condition",
            "shallow": "Shallow Flow",
            "deep": "Deep Flow",
            "sds": "SDS (cm)",
            "tau": "τ (μs)",
            "g2": "g₂",
        }

        widths = {
            "condition": 90,
            "shallow": 115,
            "deep": 115,
            "sds": 85,
            "tau": 90,
            "g2": 130,
        }

        for column in columns:
            self.human_dataset_table.heading(
                column,
                text=headings[column],
            )
            self.human_dataset_table.column(
                column,
                width=widths[column],
                minwidth=70,
                anchor="center",
                stretch=True,
            )

        y_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.human_dataset_table.yview,
        )
        x_scroll = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.human_dataset_table.xview,
        )

        self.human_dataset_table.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )

        self.human_dataset_table.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        self.human_dataset_table.bind(
            "<<TreeviewSelect>>",
            self.on_human_dataset_row_selected,
        )

        # ====================================================
        # GRAPH CONTROLS
        # ====================================================

        graph_header = ttk.Frame(right)
        graph_header.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        graph_header.columnconfigure(0, weight=1)

        ttk.Label(
            graph_header,
            text="g₂ Graph",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.human_preview_shallow_var = tk.StringVar()
        self.human_preview_deep_var = tk.StringVar()

        ttk.Label(
            graph_header,
            text="Shallow",
        ).grid(row=0, column=1, padx=(10, 3))

        self.human_preview_shallow_combo = ttk.Combobox(
            graph_header,
            textvariable=self.human_preview_shallow_var,
            state="readonly",
            width=8,
        )
        self.human_preview_shallow_combo.grid(row=0, column=2)

        ttk.Label(
            graph_header,
            text="Deep",
        ).grid(row=0, column=3, padx=(10, 3))

        self.human_preview_deep_combo = ttk.Combobox(
            graph_header,
            textvariable=self.human_preview_deep_var,
            state="readonly",
            width=8,
        )
        self.human_preview_deep_combo.grid(row=0, column=4)

        self.human_preview_shallow_combo.bind(
            "<<ComboboxSelected>>",
            self.update_human_preview,
        )
        self.human_preview_deep_combo.bind(
            "<<ComboboxSelected>>",
            self.update_human_preview,
        )

        graph_frame = ttk.Frame(right)
        graph_frame.grid(row=3, column=0, sticky="nsew")
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)

        self.human_figure = Figure(figsize=(9.2, 3.7), dpi=100)
        self.human_axis = self.human_figure.add_subplot(111)

        self.human_axis.set_xlabel("Delay Time τ (μs)")
        self.human_axis.set_ylabel("g₂(τ)")
        self.human_axis.set_title("Generate a dataset to preview")
        self.human_axis.grid(True, alpha=0.3)

        self.human_canvas = FigureCanvasTkAgg(
            self.human_figure,
            master=graph_frame,
        )
        self.human_canvas.draw()

        self.human_canvas.get_tk_widget().grid(
            row=0,
            column=0,
            sticky="nsew",
        )


    def _set_human_placeholder(self, entry):
        info = self.human_placeholder_entries[entry]
        info["active"] = True
        entry.configure(foreground="#888888")
        info["variable"].set(info["placeholder"])


    def _clear_human_placeholder(self, entry):
        info = self.human_placeholder_entries[entry]

        if info["active"]:
            info["variable"].set("")
            entry.configure(foreground="#000000")
            info["active"] = False


    def _restore_human_placeholder(self, entry):
        info = self.human_placeholder_entries[entry]

        if not info["variable"].get().strip():
            self._set_human_placeholder(entry)


    def _human_entry_value(self, variable):
        for entry, info in self.human_placeholder_entries.items():
            if info["variable"] is variable:
                if info["active"]:
                    return ""
                return variable.get().strip()

        return variable.get().strip()


    def run_human_dataset(self):
        try:
            sex = self.sex_var.get().strip()
            sat_text = self._human_entry_value(self.sat_var)
            skin_text = self._human_entry_value(self.skin_var)

            missing = []

            if sex not in ("Female", "Male"):
                missing.append("Sex")

            if not sat_text:
                missing.append("Thigh SAT Thickness")

            if missing:
                raise ValueError(
                    "Please complete: " + ", ".join(missing) + "."
                )

            try:
                sat = float(sat_text)
            except ValueError:
                raise ValueError(
                    "Thigh SAT Thickness must be a number, "
                    "for example 10.0 mm."
                )

            if not np.isfinite(sat):
                raise ValueError(
                    "Thigh SAT Thickness must be a finite number."
                )

            if sat <= 0:
                raise ValueError(
                    "Thigh SAT Thickness must be greater than 0 mm."
                )

            skin = None

            if skin_text:
                try:
                    skin = float(skin_text)
                except ValueError:
                    raise ValueError(
                        "Skin Thickness must be a number, "
                        "for example 1.7 mm."
                    )

                if not np.isfinite(skin):
                    raise ValueError(
                        "Skin Thickness must be a finite number."
                    )

                if skin <= 0:
                    raise ValueError(
                        "Skin Thickness must be greater than 0 mm."
                    )

            # Keep enough room for the deep/muscle layer inside
            # the current 40.5 mm simulation domain.
            fallback_skin = 1.89 if sex == "Male" else 1.65
            effective_skin = skin if skin is not None else fallback_skin
            superficial_depth = sat + effective_skin

            if superficial_depth >= 40.0:
                raise ValueError(
                    "Skin + SAT is too thick for the current Human V1 "
                    "simulation domain. Please use measurements whose "
                    "combined superficial depth is below 40 mm."
                )

            shallow = [5.0, 10.0, 20.0]
            deep = [5.0, 10.0, 20.0]

            human = HumanInput(
                sex=sex,
                sat_thickness_mm=sat,
                skin_thickness_mm=skin,
            )

            self.human_generate_button.config(state="disabled")
            self.human_export_csv_button.config(state="disabled")
            self.human_export_mat_button.config(state="disabled")
            self.human_status_var.set(
                "Running MCX and generating dataset..."
            )

            threading.Thread(
                target=self._human_dataset_worker,
                args=(human, shallow, deep),
                daemon=True,
            ).start()

        except Exception as error:
            messagebox.showerror(
                "Invalid Human Input",
                str(error),
            )


    def _human_dataset_worker(self, human, shallow, deep):
        try:
            output = generate_human_dataset(
                human=human, shallow_flows=shallow, deep_flows=deep,
                nphoton=10_000_000, export=False)
            self.root.after(0, lambda: self._human_dataset_complete(output))
        except Exception as error:
            self.root.after(
                0, lambda err=error: self._human_dataset_failed(err))


    def _human_dataset_complete(self, output):
        self.last_human_output = output
        self.human_dataset = output["dataset"]

        dataset = self.human_dataset
        geometry = output["geometry"]
        layer_info = output["layer_info"]

        shallow_values = sorted(
            dataset["flow_shallow"].unique()
        )
        deep_values = sorted(
            dataset["flow_deep"].unique()
        )

        shallow_display = [
            format_flow_value(value)
            for value in shallow_values
        ]
        deep_display = [
            format_flow_value(value)
            for value in deep_values
        ]

        self.human_preview_shallow_combo["values"] = shallow_display
        self.human_preview_deep_combo["values"] = deep_display

        self.human_preview_shallow_var.set(shallow_display[0])
        self.human_preview_deep_var.set(deep_display[0])

        conditions = len(shallow_values) * len(deep_values)
        curves = conditions * len(
            sorted(dataset["sds_cm"].unique())
        )

        self.human_subject_var.set(
            f"Subject: {geometry.sex}  |  "
            f"SAT {geometry.sat_thickness_mm:.2f} mm"
        )

        self.human_geometry_var.set(
            f"Skin: {geometry.skin_thickness_mm:.2f} mm "
            f"({geometry.skin_thickness_source})  |  "
            f"Depth: {geometry.superficial_thickness_mm:.2f} mm "
            f"→ voxelized {layer_info['shallow_depth_mm']:.2f} mm"
        )

        self.human_dataset_var.set(
            f"Dataset: {conditions} conditions, "
            f"{curves} curves, "
            f"{len(dataset):,} rows"
        )

        self.human_status_var.set("Complete")

        self.human_generate_button.config(state="normal")
        self.human_export_csv_button.config(state="normal")
        self.human_export_mat_button.config(state="normal")

        self.populate_human_dataset_table()
        self.update_human_preview()


    def _human_dataset_failed(self, error):
        self.human_status_var.set("Error")
        self.human_generate_button.config(state="normal")
        self.human_export_csv_button.config(state="disabled")
        self.human_export_mat_button.config(state="disabled")
        messagebox.showerror(
            "Human Dataset Error",
            str(error),
        )


    def populate_human_dataset_table(self):
        if self.human_dataset is None:
            return

        for item in self.human_dataset_table.get_children():
            self.human_dataset_table.delete(item)

        dataset = self.human_dataset.sort_values(
            [
                "flow_shallow",
                "flow_deep",
                "sds_cm",
                "tau_us",
            ]
        )

        condition_map = {}
        condition_number = 0

        for _, row in dataset.iterrows():
            key = (
                float(row["flow_shallow"]),
                float(row["flow_deep"]),
            )

            if key not in condition_map:
                condition_number += 1
                condition_map[key] = condition_number

            self.human_dataset_table.insert(
                "",
                "end",
                values=(
                    condition_map[key],
                    format_flow_value(
                        row["flow_shallow"]
                    ),
                    format_flow_value(
                        row["flow_deep"]
                    ),
                    f"{row['sds_cm']:.0f}",
                    f"{row['tau_us']:.0f}",
                    f"{row['g2']:.8f}",
                ),
            )


    def on_human_dataset_row_selected(self, event=None):
        selection = self.human_dataset_table.selection()

        if not selection:
            return

        values = self.human_dataset_table.item(
            selection[0],
            "values",
        )

        if not values:
            return

        self.human_preview_shallow_var.set(
            str(values[1])
        )
        self.human_preview_deep_var.set(
            str(values[2])
        )

        self.update_human_preview()


    def update_human_preview(self, event=None):
        if self.human_dataset is None:
            return

        try:
            shallow = float(
                self.human_preview_shallow_var.get()
            )
            deep = float(
                self.human_preview_deep_var.get()
            )

            dataset = self.human_dataset

            condition = dataset.loc[
                np.isclose(
                    dataset["flow_shallow"],
                    shallow,
                )
                & np.isclose(
                    dataset["flow_deep"],
                    deep,
                )
            ]

            if condition.empty:
                raise ValueError(
                    "Selected flow condition was not found."
                )

            self.human_axis.clear()

            for sds in sorted(
                condition["sds_cm"].unique()
            ):
                curve = condition.loc[
                    condition["sds_cm"] == sds
                ].sort_values("tau_us")

                self.human_axis.plot(
                    curve["tau_us"].to_numpy(),
                    curve["g2"].to_numpy(),
                    linewidth=2.0,
                    label=f"SDS {sds:g} cm",
                )

            self.human_axis.set_title(
                f"Human Synthetic g₂  |  "
                f"Shallow Flow = {shallow:g}, "
                f"Deep Flow = {deep:g}",
                fontsize=12,
                pad=12,
            )

            self.human_axis.set_xlabel(
                "Delay Time τ (μs)"
            )
            self.human_axis.set_ylabel(
                "g₂(τ)"
            )

            self.human_axis.grid(
                True,
                alpha=0.3,
            )

            self.human_axis.legend(
                loc="upper right",
                ncol=3,
                fontsize=9,
                frameon=True,
            )

            tau_min = float(
                condition["tau_us"].min()
            )
            tau_max = float(
                condition["tau_us"].max()
            )

            self.human_axis.set_xlim(
                tau_min,
                tau_max,
            )

            self.human_axis.margins(x=0)

            self.human_figure.subplots_adjust(
                left=0.085,
                right=0.985,
                bottom=0.18,
                top=0.88,
            )

            self.human_canvas.draw()

        except Exception as error:
            messagebox.showerror(
                "Preview Error",
                str(error),
            )


    def export_human_dataset_csv(self):
        if self.human_dataset is None:
            messagebox.showinfo(
                "Download Dataset",
                "Generate a dataset first.",
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Download Human DCS Dataset as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="human_dcs_dataset.csv",
        )

        if not filename:
            return

        try:
            self.human_dataset.to_csv(filename, index=False)
            messagebox.showinfo(
                "Download Complete",
                f"CSV dataset saved to:\n{filename}",
            )
        except Exception as error:
            messagebox.showerror("CSV Export Error", str(error))


    def export_human_dataset_mat(self):
        if self.human_dataset is None or self.last_human_output is None:
            messagebox.showinfo(
                "Download Dataset",
                "Generate a dataset first.",
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Download Human DCS Dataset as MAT",
            defaultextension=".mat",
            filetypes=[("MATLAB MAT files", "*.mat")],
            initialfile="human_dcs_dataset.mat",
        )

        if not filename:
            return

        try:
            dataset = self.human_dataset
            geometry = self.last_human_output["geometry"]
            layer_info = self.last_human_output["layer_info"]

            # Store both the numerical dataset and useful metadata.
            # Strings are stored separately because MATLAB handles
            # mixed-type table-like data differently from pandas.
            mat_data = {
                "sex": str(geometry.sex),
                "skin_thickness_mm": float(geometry.skin_thickness_mm),
                "skin_thickness_source": str(geometry.skin_thickness_source),
                "sat_thickness_mm": float(geometry.sat_thickness_mm),
                "requested_superficial_depth_mm": float(
                    geometry.superficial_thickness_mm
                ),
                "voxelized_superficial_depth_mm": float(
                    layer_info["shallow_depth_mm"]
                ),
                "deep_region_depth_mm": float(
                    layer_info["deep_depth_mm"]
                ),
                "flow_shallow": dataset["flow_shallow"].to_numpy(dtype=float),
                "flow_deep": dataset["flow_deep"].to_numpy(dtype=float),
                "alpha_db_shallow_mm2_s": dataset[
                    "alpha_db_shallow_mm2_s"
                ].to_numpy(dtype=float),
                "alpha_db_deep_mm2_s": dataset[
                    "alpha_db_deep_mm2_s"
                ].to_numpy(dtype=float),
                "sds_cm": dataset["sds_cm"].to_numpy(dtype=float),
                "tau_us": dataset["tau_us"].to_numpy(dtype=float),
                "g2": dataset["g2"].to_numpy(dtype=float),
                "column_names": np.array(
                    list(dataset.columns),
                    dtype=object,
                ),
                "generator_version": "Human DCS Simulator V1",
                "notes": (
                    "Synthetic forward-model dataset. "
                    "Flow values are assumed simulation conditions; "
                    "this file does not estimate blood flow from measured g2."
                ),
            }

            savemat(filename, mat_data)

            messagebox.showinfo(
                "Download Complete",
                f"MAT dataset saved to:\n{filename}",
            )

        except Exception as error:
            messagebox.showerror("MAT Export Error", str(error))

# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    DCSSimulatorGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()