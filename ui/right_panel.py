import customtkinter as ctk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class RightPanel(ctk.CTkFrame):

    def __init__(self, parent):

        super().__init__(parent)

        self.graph_canvas = None
        self.graph_figure = None

        self.dataset_rows = []
        self.selected_trial = None

        self.grid_propagate(False)

        # ==========================
        # Right Panel Grid
        # ==========================

        self.grid_columnconfigure(
            0,
            weight=1
        )

        # Dataset Browser สูงขึ้น
        self.grid_rowconfigure(
            1,
            weight=0,
            minsize=330
        )

        # Graph ใช้พื้นที่ที่เหลือ
        self.grid_rowconfigure(
            3,
            weight=1,
            minsize=280
        )

        # ==========================
        # Dataset Browser Title
        # ==========================

        dataset_title = ctk.CTkLabel(
            self,
            text="Dataset Browser",
            font=("Arial", 24, "bold")
        )

        dataset_title.grid(
            row=0,
            column=0,
            sticky="w",
            padx=20,
            pady=(20, 10)
        )

        # ==========================
        # Dataset Textbox
        # ==========================

        self.dataset_textbox = ctk.CTkTextbox(
            self,
            height=330,
            wrap="none",
            font=("Courier New", 12)
        )

        self.dataset_textbox.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 12)
        )

        self.dataset_textbox.insert(
            "1.0",
            "No dataset loaded."
        )

        self.dataset_textbox.configure(
            state="disabled"
        )

        # ==========================
        # Graph Title
        # ==========================

        graph_title = ctk.CTkLabel(
            self,
            text="Graph",
            font=("Arial", 24, "bold")
        )

        graph_title.grid(
            row=2,
            column=0,
            sticky="w",
            padx=20,
            pady=(0, 8)
        )

        # ==========================
        # Graph Frame
        # ==========================

        self.graph_frame = ctk.CTkFrame(
            self
        )

        self.graph_frame.grid(
            row=3,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 20)
        )

        self.graph_frame.grid_propagate(False)
        self.graph_frame.pack_propagate(False)

        self.show_graph_placeholder(
            "No graph to display."
        )

    # ==================================
    # Show Entire Dataset
    # ==================================

    def show_dataset(
        self,
        dataset_rows
    ):

        self.dataset_rows = dataset_rows
        self.selected_trial = None

        self.refresh_dataset_text()

    # ==================================
    # Build Dataset Text
    # ==================================

    def refresh_dataset_text(self):

        if not self.dataset_rows:

            dataset_text = "No dataset loaded."

        else:

            header = (
                " Sel  Trial  "
                "Preset S  Preset D  "
                "Measured S  Measured D  "
                "BFI 1 cm    BFI 2 cm    BFI 3 cm\n"
            )

            separator = (
                "-" * 112
                + "\n"
            )

            lines = [
                header,
                separator
            ]

            for row in self.dataset_rows:

                if row["trial"] == self.selected_trial:
                    selected_marker = ">>>"
                else:
                    selected_marker = "   "

                line = (
                    f"{selected_marker}  "
                    f"{row['trial']:>5}  "
                    f"{row['preset_shallow_flow']:>8.2f}  "
                    f"{row['preset_deep_flow']:>8.2f}  "
                    f"{row['measured_shallow_flow']:>10.3f}  "
                    f"{row['measured_deep_flow']:>10.3f}  "
                    f"{row['physical_bfi_1cm']:>10.3f}  "
                    f"{row['physical_bfi_2cm']:>10.3f}  "
                    f"{row['physical_bfi_3cm']:>10.3f}\n"
                )

                lines.append(line)

            dataset_text = "".join(lines)

        self.dataset_textbox.configure(
            state="normal"
        )

        self.dataset_textbox.delete(
            "1.0",
            "end"
        )

        self.dataset_textbox.insert(
            "1.0",
            dataset_text
        )

        self.dataset_textbox.configure(
            state="disabled"
        )

    # ==================================
    # Highlight Selected Trial
    # ==================================

    def highlight_dataset_row(
        self,
        trial
    ):

        self.selected_trial = int(
            trial
        )

        self.refresh_dataset_text()

        # Header กับเส้นคั่นมีทั้งหมด 2 บรรทัด
        selected_line_number = (
            self.selected_trial + 2
        )

        self.dataset_textbox.configure(
            state="normal"
        )

        self.dataset_textbox.see(
            f"{selected_line_number}.0"
        )

        self.dataset_textbox.configure(
            state="disabled"
        )

    # ==================================
    # Clear Selected Trial
    # ==================================

    def clear_selected_row(self):

        self.selected_trial = None

        if self.dataset_rows:
            self.refresh_dataset_text()

    # ==================================
    # Clear Graph
    # ==================================

    def clear_graph_frame(self):

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        if self.graph_figure is not None:
            self.graph_figure.clear()

        self.graph_figure = None
        self.graph_canvas = None

    # ==================================
    # Graph Placeholder
    # ==================================

    def show_graph_placeholder(
        self,
        message
    ):

        self.clear_graph_frame()

        placeholder = ctk.CTkLabel(
            self.graph_frame,
            text=message,
            text_color="gray",
            font=("Arial", 18)
        )

        placeholder.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

    # ==================================
    # Show Graph
    # ==================================

    def show_graph(
        self,
        result
    ):

        self.clear_graph_frame()

        # ใช้ขนาดคงที่ที่เล็กกว่าช่องขวา
        # ไม่ต้องให้ Tkinter หรือ Windows scaling เดาขนาด
        self.graph_figure = Figure(
            figsize=(9.0, 3.1),
            dpi=100
        )

        axis = self.graph_figure.add_subplot(
            111
        )

        axis.plot(
            result["tau"],
            result["g2_1cm"],
            label="SDS 1 cm"
        )

        axis.plot(
            result["tau"],
            result["g2_2cm"],
            label="SDS 2 cm"
        )

        axis.plot(
            result["tau"],
            result["g2_3cm"],
            label="SDS 3 cm"
        )

        axis.set_title(
            f"g2 Curves - Trial {result['trial']}",
            fontsize=12,
            pad=28
        )

        axis.set_xlabel(
            "Delay Time (µs)",
            fontsize=10
        )

        axis.set_ylabel(
            "g2",
            fontsize=10
        )

        axis.tick_params(
            axis="both",
            labelsize=9
        )

        axis.grid(
            True,
            alpha=0.3
        )

        # Legend แนวนอนอยู่ด้านบน
        # ไม่กินขอบขวาอีก
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            fontsize=9,
            frameon=True
        )

        axis.set_xlim(
            float(result["tau"][0]),
            float(result["tau"][-1])
        )

        axis.margins(
            x=0
        )

        # เผื่อขอบครบทุกด้าน
        self.graph_figure.subplots_adjust(
            left=0.10,
            right=0.97,
            bottom=0.20,
            top=0.76
        )

        self.graph_canvas = FigureCanvasTkAgg(
            self.graph_figure,
            master=self.graph_frame
        )

        self.graph_canvas.draw()

        canvas_widget = (
            self.graph_canvas.get_tk_widget()
        )

        canvas_widget.configure(
            width=900,
            height=310,
            highlightthickness=0,
            borderwidth=0
        )

        # ไม่ขยายเต็มช่อง
        # วางกราฟขนาดคงที่ตรงกลางแทน
        canvas_widget.pack(
            anchor="center",
            pady=5
        )