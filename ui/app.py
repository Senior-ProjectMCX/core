import customtkinter as ctk

from ui.left_panel import LeftPanel
from ui.right_panel import RightPanel
from ui.pipeline import Pipeline


class App(ctk.CTk):

    def __init__(self):

        super().__init__()

        # ==========================
        # Window Settings
        # ==========================

        self.title("DCS Dataset Explorer")

        self.geometry("1400x800")
        self.minsize(1400, 800)
        self.maxsize(1400, 800)

        # ล็อกขนาดหน้าต่าง
        # ไม่ให้ Matplotlib หรือ Widget อื่นดันหน้าต่างให้ใหญ่ขึ้น
        self.resizable(False, False)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # ==========================
        # Main Grid
        # ==========================

        self.grid_columnconfigure(
            0,
            weight=0,
            minsize=360
        )

        self.grid_columnconfigure(
            1,
            weight=1
        )

        self.grid_rowconfigure(
            0,
            weight=1
        )

        # ==========================
        # Pipeline
        # ==========================

        self.pipeline = Pipeline()

        # ==========================
        # Right Panel
        # ==========================

        self.right = RightPanel(self)

        self.right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(0, 10),
            pady=10
        )

        # ==========================
        # Left Panel
        # ==========================

        self.left = LeftPanel(self)

        self.left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10
        )

        # ==========================
        # Connect Components
        # ==========================

        self.left.pipeline = self.pipeline
        self.left.right_panel = self.right