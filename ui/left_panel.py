import queue
import threading
import tkinter.messagebox as messagebox

import customtkinter as ctk


class LeftPanel(ctk.CTkFrame):

    SELECT_TEXT = "Select a value"
    SELECT_TRIAL_TEXT = "Select a trial"

    def __init__(self, parent):

        super().__init__(
            parent,
            width=320
        )

        self.pack_propagate(False)

        self.pipeline = None
        self.right_panel = None

        self.updating_dropdowns = False

        self.worker_queue = queue.Queue()
        self.generation_thread = None

        # Progress แบบลื่น
        self.displayed_progress = 0.0
        self.target_progress = 0.0
        self.progress_animation_running = False

        # ==========================
        # Title
        # ==========================

        title = ctk.CTkLabel(
            self,
            text="DCS Dataset Explorer",
            font=("Arial", 24, "bold")
        )

        title.pack(
            pady=20
        )

        # ==========================
        # Generate Dataset
        # ==========================

        self.generate_button = ctk.CTkButton(
            self,
            text="Generate Dataset",
            command=self.generate_dataset
        )

        self.generate_button.pack(
            fill="x",
            padx=20,
            pady=(10, 15)
        )

        # ==========================
        # Status
        # ==========================

        self.status_label = ctk.CTkLabel(
            self,
            text="No dataset loaded.",
            text_color="gray"
        )

        self.status_label.pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        # ==========================
        # Progress Bar
        # ==========================

        self.progress_bar = ctk.CTkProgressBar(
            self
        )

        self.progress_bar.pack(
            fill="x",
            padx=20,
            pady=(0, 5)
        )

        self.progress_bar.set(0)

        self.progress_text = ctk.CTkLabel(
            self,
            text="0%",
            text_color="gray"
        )

        self.progress_text.pack(
            anchor="e",
            padx=20,
            pady=(0, 20)
        )

        # ==========================
        # Shallow Flow
        # ==========================

        ctk.CTkLabel(
            self,
            text="Shallow Flow"
        ).pack(
            anchor="w",
            padx=20
        )

        self.shallow_combobox = ctk.CTkComboBox(
            self,
            values=[self.SELECT_TEXT],
            command=self.on_shallow_selected,
            state="disabled"
        )

        self.shallow_combobox.pack(
            fill="x",
            padx=20,
            pady=(5, 15)
        )

        self.shallow_combobox.set(
            self.SELECT_TEXT
        )

        # ==========================
        # Deep Flow
        # ==========================

        ctk.CTkLabel(
            self,
            text="Deep Flow"
        ).pack(
            anchor="w",
            padx=20
        )

        self.deep_combobox = ctk.CTkComboBox(
            self,
            values=[self.SELECT_TEXT],
            command=self.on_deep_selected,
            state="disabled"
        )

        self.deep_combobox.pack(
            fill="x",
            padx=20,
            pady=(5, 15)
        )

        self.deep_combobox.set(
            self.SELECT_TEXT
        )

        # ==========================
        # Matching Trial
        # ==========================

        ctk.CTkLabel(
            self,
            text="Matching Trial"
        ).pack(
            anchor="w",
            padx=20
        )

        self.trial_combobox = ctk.CTkComboBox(
            self,
            values=[self.SELECT_TRIAL_TEXT],
            command=self.on_trial_selected,
            state="disabled"
        )

        self.trial_combobox.pack(
            fill="x",
            padx=20,
            pady=(5, 20)
        )

        self.trial_combobox.set(
            self.SELECT_TRIAL_TEXT
        )

        # ==========================
        # Reset Button
        # ==========================

        self.reset_button = ctk.CTkButton(
            self,
            text="Reset Selection",
            command=self.reset_selection,
            state="disabled"
        )

        self.reset_button.pack(
            fill="x",
            padx=20,
            pady=10
        )

        # ==========================
        # Hint
        # ==========================

        self.hint_label = ctk.CTkLabel(
            self,
            text=(
                "Select Shallow Flow and Deep Flow.\n"
                "Then choose one matching trial."
            ),
            text_color="gray",
            justify="left"
        )

        self.hint_label.pack(
            anchor="w",
            padx=20,
            pady=(10, 0)
        )

    # ==================================
    # Number Formatting
    # ==================================

    def format_flow_value(
        self,
        value
    ):

        value = float(value)

        if value.is_integer():
            return str(int(value))

        return f"{value:g}"

    def format_flow_values(
        self,
        values
    ):

        return [
            self.format_flow_value(value)
            for value in values
        ]

    # ==================================
    # Generate Dataset
    # ==================================

    def generate_dataset(self):

        if self.pipeline is None:

            messagebox.showerror(
                "Pipeline Error",
                "Pipeline is not connected."
            )

            return

        if (
            self.generation_thread is not None
            and self.generation_thread.is_alive()
        ):
            return

        self.generate_button.configure(
            state="disabled",
            text="Generating..."
        )

        self.shallow_combobox.configure(
            state="disabled"
        )

        self.deep_combobox.configure(
            state="disabled"
        )

        self.trial_combobox.configure(
            state="disabled"
        )

        self.reset_button.configure(
            state="disabled"
        )

        self.status_label.configure(
            text="Preparing dataset generation...",
            text_color="orange"
        )

        self.displayed_progress = 0.0
        self.target_progress = 0.0

        self.progress_bar.set(0)

        self.progress_text.configure(
            text="0%"
        )

        self.generation_thread = threading.Thread(
            target=self.run_generation_worker,
            daemon=True
        )

        self.generation_thread.start()

        self.after(
            100,
            self.process_worker_queue
        )

        if not self.progress_animation_running:

            self.progress_animation_running = True

            self.after(
                30,
                self.animate_progress
            )

    # ==================================
    # Background Worker
    # ==================================

    def run_generation_worker(self):

        try:

            number_of_samples = (
                self.pipeline.generate_dataset(
                    progress_callback=(
                        self.report_progress
                    )
                )
            )

            dataset_rows = (
                self.pipeline.get_all_dataset_rows()
            )

            shallow_values = (
                self.pipeline.get_all_shallow_values()
            )

            deep_values = (
                self.pipeline.get_all_deep_values()
            )

            self.worker_queue.put((
                "success",
                number_of_samples,
                dataset_rows,
                shallow_values,
                deep_values
            ))

        except Exception as error:

            self.worker_queue.put((
                "error",
                str(error)
            ))

    # ==================================
    # Report Progress
    # ==================================

    def report_progress(
        self,
        message,
        percentage
    ):

        self.worker_queue.put((
            "progress",
            message,
            float(percentage)
        ))

    # ==================================
    # Process Worker Queue
    # ==================================

    def process_worker_queue(self):

        generation_finished = False

        try:

            while True:

                item = (
                    self.worker_queue.get_nowait()
                )

                message_type = item[0]

                if message_type == "progress":

                    self.receive_progress(
                        message=item[1],
                        percentage=item[2]
                    )

                elif message_type == "success":

                    self.finish_generation(
                        number_of_samples=item[1],
                        dataset_rows=item[2],
                        shallow_values=item[3],
                        deep_values=item[4]
                    )

                    generation_finished = True

                elif message_type == "error":

                    self.fail_generation(
                        item[1]
                    )

                    generation_finished = True

        except queue.Empty:
            pass

        if not generation_finished:

            if (
                self.generation_thread is not None
                and self.generation_thread.is_alive()
            ):

                self.after(
                    100,
                    self.process_worker_queue
                )

            elif not self.worker_queue.empty():

                self.after(
                    100,
                    self.process_worker_queue
                )

    # ==================================
    # Receive Progress Target
    # ==================================

    def receive_progress(
        self,
        message,
        percentage
    ):

        self.status_label.configure(
            text=message,
            text_color="orange"
        )

        self.target_progress = max(
            self.target_progress,
            min(
                float(percentage),
                100.0
            )
        )

    # ==================================
    # Smooth Progress Animation
    # ==================================

    def animate_progress(self):

        difference = (
            self.target_progress
            - self.displayed_progress
        )

        if difference > 0:

            # ยิ่งเป้าหมายอยู่ไกลยิ่งขยับเร็ว
            # แต่ช่วงท้ายจะค่อย ๆ เข้าใกล้แบบลื่น
            step = max(
                0.12,
                difference * 0.045
            )

            self.displayed_progress = min(
                self.displayed_progress + step,
                self.target_progress
            )

            self.progress_bar.set(
                self.displayed_progress / 100
            )

            self.progress_text.configure(
                text=(
                    f"{self.displayed_progress:.0f}%"
                )
            )

        generation_active = (
            self.generation_thread is not None
            and self.generation_thread.is_alive()
        )

        still_animating = (
            self.displayed_progress
            < self.target_progress - 0.01
        )

        if generation_active or still_animating:

            self.after(
                30,
                self.animate_progress
            )

        else:

            self.progress_animation_running = False

    # ==================================
    # Generation Success
    # ==================================

    def finish_generation(
        self,
        number_of_samples,
        dataset_rows,
        shallow_values,
        deep_values
    ):

        shallow_text_values = (
            self.format_flow_values(
                shallow_values
            )
        )

        deep_text_values = (
            self.format_flow_values(
                deep_values
            )
        )

        self.updating_dropdowns = True

        self.shallow_combobox.configure(
            values=shallow_text_values,
            state="readonly"
        )

        self.deep_combobox.configure(
            values=deep_text_values,
            state="readonly"
        )

        self.trial_combobox.configure(
            values=[self.SELECT_TRIAL_TEXT],
            state="disabled"
        )

        self.shallow_combobox.set(
            self.SELECT_TEXT
        )

        self.deep_combobox.set(
            self.SELECT_TEXT
        )

        self.trial_combobox.set(
            self.SELECT_TRIAL_TEXT
        )

        self.updating_dropdowns = False

        self.reset_button.configure(
            state="normal"
        )

        self.generate_button.configure(
            state="normal",
            text="Generate Dataset"
        )

        self.status_label.configure(
            text=(
                f"Dataset loaded "
                f"({number_of_samples} samples)"
            ),
            text_color="green"
        )

        self.target_progress = 100.0
        self.displayed_progress = 100.0

        self.progress_bar.set(1)

        self.progress_text.configure(
            text="100%"
        )

        if self.right_panel is not None:

            self.right_panel.show_dataset(
                dataset_rows
            )

            self.right_panel.show_graph_placeholder(
                "Select flow values and a matching trial."
            )

    # ==================================
    # Generation Failed
    # ==================================

    def fail_generation(
        self,
        error_message
    ):

        self.generate_button.configure(
            state="normal",
            text="Generate Dataset"
        )

        self.status_label.configure(
            text="Dataset generation failed.",
            text_color="red"
        )

        self.target_progress = 0.0
        self.displayed_progress = 0.0

        self.progress_bar.set(0)

        self.progress_text.configure(
            text="Failed"
        )

        messagebox.showerror(
            "Dataset Error",
            error_message
        )

    # ==================================
    # Shallow Selected
    # ==================================

    def on_shallow_selected(
        self,
        selected_value
    ):

        if self.updating_dropdowns:
            return

        if (
            not selected_value
            or selected_value == self.SELECT_TEXT
        ):
            return

        try:

            compatible_deep_values = (
                self.pipeline
                .get_deep_values_for_shallow(
                    selected_value
                )
            )

            deep_text_values = (
                self.format_flow_values(
                    compatible_deep_values
                )
            )

            current_deep = (
                self.deep_combobox.get()
            )

            self.updating_dropdowns = True

            self.deep_combobox.configure(
                values=deep_text_values,
                state="readonly"
            )

            if current_deep not in deep_text_values:

                self.deep_combobox.set(
                    self.SELECT_TEXT
                )

            self.clear_trial_dropdown()

            self.updating_dropdowns = False

            self.update_matching_trials()

        except Exception as error:

            self.updating_dropdowns = False

            messagebox.showerror(
                "Dropdown Error",
                str(error)
            )

    # ==================================
    # Deep Selected
    # ==================================

    def on_deep_selected(
        self,
        selected_value
    ):

        if self.updating_dropdowns:
            return

        if (
            not selected_value
            or selected_value == self.SELECT_TEXT
        ):
            return

        try:

            compatible_shallow_values = (
                self.pipeline
                .get_shallow_values_for_deep(
                    selected_value
                )
            )

            shallow_text_values = (
                self.format_flow_values(
                    compatible_shallow_values
                )
            )

            current_shallow = (
                self.shallow_combobox.get()
            )

            self.updating_dropdowns = True

            self.shallow_combobox.configure(
                values=shallow_text_values,
                state="readonly"
            )

            if (
                current_shallow
                not in shallow_text_values
            ):

                self.shallow_combobox.set(
                    self.SELECT_TEXT
                )

            self.clear_trial_dropdown()

            self.updating_dropdowns = False

            self.update_matching_trials()

        except Exception as error:

            self.updating_dropdowns = False

            messagebox.showerror(
                "Dropdown Error",
                str(error)
            )

    # ==================================
    # Clear Trial Dropdown
    # ==================================

    def clear_trial_dropdown(self):

        self.trial_combobox.configure(
            values=[self.SELECT_TRIAL_TEXT],
            state="disabled"
        )

        self.trial_combobox.set(
            self.SELECT_TRIAL_TEXT
        )

        if self.right_panel is not None:

            self.right_panel.clear_selected_row()

            self.right_panel.show_graph_placeholder(
                "Select a matching trial."
            )

    # ==================================
    # Update Matching Trials
    # ==================================

    def update_matching_trials(self):

        shallow_value = (
            self.shallow_combobox.get()
        )

        deep_value = (
            self.deep_combobox.get()
        )

        if (
            not shallow_value
            or not deep_value
            or shallow_value == self.SELECT_TEXT
            or deep_value == self.SELECT_TEXT
        ):
            return

        matching_trials = (
            self.pipeline.get_matching_trials(
                shallow_flow=float(
                    shallow_value
                ),
                deep_flow=float(
                    deep_value
                )
            )
        )

        trial_values = [
            f"Trial {trial}"
            for trial in matching_trials
        ]

        self.updating_dropdowns = True

        if trial_values:

            self.trial_combobox.configure(
                values=trial_values,
                state="readonly"
            )

            self.trial_combobox.set(
                self.SELECT_TRIAL_TEXT
            )

            self.status_label.configure(
                text=(
                    f"{len(trial_values)} matching "
                    "trials found."
                ),
                text_color="green"
            )

        else:

            self.trial_combobox.configure(
                values=[self.SELECT_TRIAL_TEXT],
                state="disabled"
            )

            self.trial_combobox.set(
                self.SELECT_TRIAL_TEXT
            )

            self.status_label.configure(
                text="No matching trials found.",
                text_color="red"
            )

        self.updating_dropdowns = False

    # ==================================
    # Trial Selected
    # ==================================

    def on_trial_selected(
        self,
        selected_value
    ):

        if self.updating_dropdowns:
            return

        if (
            not selected_value
            or selected_value
            == self.SELECT_TRIAL_TEXT
        ):
            return

        try:

            trial = int(
                selected_value.replace(
                    "Trial",
                    ""
                ).strip()
            )

            result = (
                self.pipeline.get_trial_result(
                    trial
                )
            )

            if self.right_panel is not None:

                self.right_panel.highlight_dataset_row(
                    trial
                )

                self.right_panel.show_graph(
                    result
                )

            self.status_label.configure(
                text=f"Showing Trial {trial}.",
                text_color="green"
            )

        except Exception as error:

            messagebox.showerror(
                "Trial Error",
                str(error)
            )

    # ==================================
    # Reset Selection
    # ==================================

    def reset_selection(self):

        if (
            self.pipeline is None
            or not self.pipeline.is_dataset_loaded()
        ):
            return

        shallow_values = (
            self.pipeline.get_all_shallow_values()
        )

        deep_values = (
            self.pipeline.get_all_deep_values()
        )

        self.updating_dropdowns = True

        self.shallow_combobox.configure(
            values=self.format_flow_values(
                shallow_values
            ),
            state="readonly"
        )

        self.deep_combobox.configure(
            values=self.format_flow_values(
                deep_values
            ),
            state="readonly"
        )

        self.trial_combobox.configure(
            values=[self.SELECT_TRIAL_TEXT],
            state="disabled"
        )

        self.shallow_combobox.set(
            self.SELECT_TEXT
        )

        self.deep_combobox.set(
            self.SELECT_TEXT
        )

        self.trial_combobox.set(
            self.SELECT_TRIAL_TEXT
        )

        self.updating_dropdowns = False

        self.status_label.configure(
            text=(
                f"Dataset loaded "
                f"({self.pipeline.get_dataset_size()} samples)"
            ),
            text_color="green"
        )

        if self.right_panel is not None:

            self.right_panel.clear_selected_row()

            self.right_panel.show_graph_placeholder(
                "Select flow values and a matching trial."
            )