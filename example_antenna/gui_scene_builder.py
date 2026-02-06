import os
import sys
import yaml
import queue
import threading
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk  # type: ignore
    PIL_OK = True
except Exception:
    PIL_OK = False


# ----------------------------
# MATERIALS (ALL)
# ----------------------------
DEFAULT_MATERIALS = {
    "air":      {"absorption": 0.05, "R": 0.00, "T": 1.00, "scatter": 0.00},
    "metal":    {"absorption": 5.00, "R": 0.90, "T": 0.10, "scatter": 1.00},
    "plastic":  {"absorption": 0.35, "R": 0.10, "T": 0.90, "scatter": 0.18},
    "glass":    {"absorption": 0.60, "R": 0.15, "T": 0.85, "scatter": 0.25},
    "concrete": {"absorption": 2.50, "R": 0.60, "T": 0.40, "scatter": 0.55},
    "water":    {"absorption": 0.50, "R": 0.10, "T": 0.90, "scatter": 0.25},
    "wood":     {"absorption": 1.00, "R": 0.30, "T": 0.70, "scatter": 0.35},
    "rubber":   {"absorption": 3.00, "R": 0.25, "T": 0.75, "scatter": 0.20},
    "ice":      {"absorption": 0.20, "R": 0.08, "T": 0.92, "scatter": 0.12},
    "sand":     {"absorption": 1.80, "R": 0.35, "T": 0.65, "scatter": 0.45},
    "brick":    {"absorption": 2.20, "R": 0.55, "T": 0.45, "scatter": 0.50},
    "asphalt":  {"absorption": 3.20, "R": 0.50, "T": 0.50, "scatter": 0.40},
    "foam":     {"absorption": 4.50, "R": 0.10, "T": 0.90, "scatter": 0.08},
}

# colors for preview (nice palette)
PALETTE = [
    "#22c55e", "#60a5fa", "#f59e0b", "#a78bfa", "#f472b6",
    "#34d399", "#fb7185", "#c084fc", "#38bdf8", "#facc15",
    "#4ade80", "#e879f9", "#2dd4bf", "#f97316",
]
MATERIAL_COLORS = {
    "air": "#94a3b8",
    "metal": "#eab308",
    "plastic": "#60a5fa",
    "glass": "#38bdf8",
    "concrete": "#a3a3a3",
    "water": "#3b82f6",
    "wood": "#b45309",
    "rubber": "#111827",
    "ice": "#93c5fd",
    "sand": "#fbbf24",
    "brick": "#ef4444",
    "asphalt": "#374151",
    "foam": "#a7f3d0",
}

DOMAIN = dict(xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)
SOURCE = dict(x0=0.8, y0=1.0, amplitude=1.0)


def ffloat(s: str, name: str) -> float:
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"'{name}' must be a number. Current value: {s!r}")


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


def open_in_file_manager(path: Path):
    path = path.resolve()
    if path.is_file():
        target_dir = path.parent
        select_file = path
    else:
        target_dir = path
        select_file = None

    try:
        if sys.platform.startswith("win"):
            if select_file:
                subprocess.Popen(["explorer.exe", "/select,", str(select_file)])
            else:
                os.startfile(str(target_dir))  # type: ignore[attr-defined]
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
            return

        if _is_wsl():
            win_dir = subprocess.check_output(["wslpath", "-w", str(target_dir)], text=True).strip()
            if select_file:
                win_file = subprocess.check_output(["wslpath", "-w", str(select_file)], text=True).strip()
                subprocess.Popen(["explorer.exe", "/select,", win_file])
            else:
                subprocess.Popen(["explorer.exe", win_dir])
            return

        if subprocess.call(["which", "xdg-open"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            subprocess.Popen(["xdg-open", str(target_dir)])
            return

        raise FileNotFoundError("xdg-open not found. Install: sudo apt install xdg-utils")
    except Exception as e:
        messagebox.showwarning("Cannot open folder", str(e))


@dataclass
class ObjRow:
    frame: ttk.Frame
    idx_lbl: ttk.Label
    type_cb: ttk.Combobox
    mat_cb: ttk.Combobox
    x1_ent: ttk.Entry
    x2_ent: ttk.Entry
    y1_ent: ttk.Entry
    y2_ent: ttk.Entry
    del_btn: ttk.Button


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EM PINN Scene Builder (Rect + Circle)")
        self.geometry("1400x900")
        self.minsize(1180, 720)

        self.project_dir = Path(__file__).resolve().parent
        self.scene_path = self.project_dir / "scene.yaml"
        self.solver_path = self.project_dir / "solver_simple_torch.py"

        self.rows: List[ObjRow] = []
        self.running = False
        self.last_results_dir: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None

        self._q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._polling = False
        self._img_refs = {}

        # placement
        self.selected_obj_idx = tk.IntVar(value=1)  # 1-based
        self.mode_var = tk.StringVar(value="Rect: set corner 1")
        self.snap_enabled = tk.BooleanVar(value=True)
        self.snap_step = tk.StringVar(value="0.05")
        self.cursor_var = tk.StringVar(value="X: -, Y: -")

        # progress
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_txt = tk.StringVar(value="0%")
        self.status_var = tk.StringVar(value="Ready")

        self._style()
        self._ui()

        self.n_var.set(1)
        self.apply_n()
        self.fill_examples()
        self.after(80, self.preview_scene)

    def _style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Danger.TButton", foreground="#b91c1c")

    def _ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        head = ttk.Frame(root)
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="EM PINN Scene Builder (Rect + Circle)", style="Header.TLabel").pack(side="left")

        right_head = ttk.Frame(head)
        right_head.pack(side="right")
        ttk.Label(right_head, textvariable=self.status_var).pack(side="right", padx=(0, 10))
        ttk.Label(right_head, textvariable=self.progress_txt).pack(side="right")
        ttk.Progressbar(
            right_head, orient="horizontal", length=240,
            mode="determinate", maximum=100.0, variable=self.progress_var
        ).pack(side="right", padx=(8, 8))

        ctrl = ttk.LabelFrame(root, text="Parameters & Controls")
        ctrl.pack(fill="x", pady=(0, 10))

        bar = ttk.Frame(ctrl)
        bar.pack(fill="x", padx=8, pady=8)

        ttk.Label(bar, text="Frequency (GHz):").pack(side="left")
        self.freq_var = tk.StringVar(value="1.0")
        ttk.Entry(bar, textvariable=self.freq_var, width=8).pack(side="left", padx=(6, 14))

        ttk.Label(bar, text="Objects (N):").pack(side="left")
        self.n_var = tk.IntVar(value=1)
        ttk.Spinbox(bar, from_=0, to=50, textvariable=self.n_var, width=5).pack(side="left", padx=(6, 8))
        ttk.Button(bar, text="Apply N", command=self.apply_n).pack(side="left", padx=(0, 14))

        ttk.Label(bar, text="Epochs:").pack(side="left")
        self.epochs_var = tk.IntVar(value=2000)
        ttk.Spinbox(bar, from_=2000, to=200000, increment=1000, textvariable=self.epochs_var, width=8)\
            .pack(side="left", padx=(6, 14))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(bar, text="Example", command=self.fill_examples).pack(side="left", padx=4)
        ttk.Button(bar, text="Preview", command=self.preview_scene).pack(side="left", padx=4)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(bar, text="Save scene.yaml", command=self.save_scene).pack(side="left", padx=4)

        self.btn_run = ttk.Button(bar, text="▶ Run solver", style="Primary.TButton", command=self.run_solver)
        self.btn_run.pack(side="left", padx=6)

        self.btn_restart = ttk.Button(bar, text="Restart", command=self.restart_app)
        self.btn_restart.pack(side="left", padx=4)

        self.btn_open = ttk.Button(bar, text="Open results folder", command=self.open_results, state="disabled")
        self.btn_open.pack(side="left", padx=4)

        self.path_var = tk.StringVar(value=f"scene.yaml: {self.scene_path}")
        ttk.Label(ctrl, textvariable=self.path_var).pack(anchor="w", padx=10, pady=(0, 8))

        # Placement
        place = ttk.LabelFrame(root, text="Placement (click on Preview)")
        place.pack(fill="x", pady=(0, 10))

        pl = ttk.Frame(place)
        pl.pack(fill="x", padx=8, pady=8)

        ttk.Label(pl, text="Selected object:").pack(side="left")
        self.sel_spin = ttk.Spinbox(pl, from_=1, to=1, textvariable=self.selected_obj_idx, width=5)
        self.sel_spin.pack(side="left", padx=(6, 16))

        ttk.Label(pl, text="Mode:").pack(side="left")
        mode_cb = ttk.Combobox(
            pl,
            values=["Rect: set corner 1", "Rect: set corner 2", "Rect: move center",
                    "Circle: set center", "Circle: set radius"],
            state="readonly",
            textvariable=self.mode_var,
            width=18
        )
        mode_cb.pack(side="left", padx=(6, 10))
        mode_cb.current(0)

        ttk.Separator(pl, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Checkbutton(pl, text="Snap", variable=self.snap_enabled).pack(side="left")
        ttk.Label(pl, text="Step:").pack(side="left", padx=(8, 4))
        snap_cb = ttk.Combobox(pl, values=["0.01", "0.05", "0.1"], state="readonly", textvariable=self.snap_step, width=6)
        snap_cb.pack(side="left", padx=(0, 12))

        # Layout
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        # Objects
        obj_box = ttk.LabelFrame(main, text="Objects (rect: x1/x2/y1/y2) | (circle: x1=cx, y1=cy, x2=r)")
        obj_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))

        hdr = ttk.Frame(obj_box)
        hdr.pack(fill="x", pady=(0, 6), padx=6)
        cols = [("#", 3), ("Type", 7), ("Material", 12), ("x1/cx", 8), ("x2/r", 8), ("y1/cy", 8), ("y2", 8), ("", 9)]
        for j, (name, w) in enumerate(cols):
            ttk.Label(hdr, text=name, width=w, anchor="center").grid(row=0, column=j, padx=3)

        self.table_canvas = tk.Canvas(obj_box, highlightthickness=0)
        self.table_scroll = ttk.Scrollbar(obj_box, orient="vertical", command=self.table_canvas.yview)
        self.table_inner = ttk.Frame(self.table_canvas)
        self.table_inner.bind("<Configure>", lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all")))
        self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")
        self.table_canvas.configure(yscrollcommand=self.table_scroll.set)
        self.table_canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        self.table_scroll.pack(side="right", fill="y", padx=(0, 6), pady=(0, 6))

        # Preview
        prev_box = ttk.LabelFrame(main, text="Preview (colored by material, red=source)")
        prev_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        self.preview = tk.Canvas(prev_box, background="#0b1220")
        self.preview.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        self.preview.bind("<Button-1>", self._on_preview_click)
        self.preview.bind("<Motion>", self._on_mouse_move)

        ttk.Label(prev_box, textvariable=self.cursor_var, anchor="w").pack(fill="x", padx=10, pady=(4, 6))

        # Right notebook
        right = ttk.Frame(main)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")

        self.tab_log = ttk.Frame(self.nb)
        self.nb.add(self.tab_log, text="Log")
        self.log = tk.Text(self.tab_log, wrap="word")
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_res = ttk.Frame(self.nb)
        self.nb.add(self.tab_res, text="Results")
        self._build_results_ui()

        self._log("Ready: Apply N → Example/Preview → click placement → Save → Run\n")
        if not PIL_OK:
            self._log("[WARN] Pillow not installed. PNG display may be limited.\nInstall: pip install pillow\n")

    def _build_results_ui(self):
        self.tab_res.columnconfigure(0, weight=1)
        self.tab_res.rowconfigure(2, weight=1)

        info = ttk.LabelFrame(self.tab_res, text="Summary")
        info.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.res_path_var = tk.StringVar(value="Results folder: (none)")
        self.res_metrics_var = tk.StringVar(value="Metrics: (none)")
        ttk.Label(info, textvariable=self.res_path_var).pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Label(info, textvariable=self.res_metrics_var).pack(anchor="w", padx=8, pady=(0, 6))

        btns = ttk.Frame(self.tab_res)
        btns.grid(row=1, column=0, sticky="ew", padx=6)
        ttk.Button(btns, text="Reload results", command=self.show_results).pack(side="left")
        ttk.Button(btns, text="Open folder", command=self.open_results).pack(side="left", padx=6)

        imgs = ttk.LabelFrame(self.tab_res, text="Field images")
        imgs.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        imgs.columnconfigure(0, weight=1)
        imgs.columnconfigure(1, weight=1)
        imgs.columnconfigure(2, weight=1)
        imgs.rowconfigure(0, weight=1)

        self.lbl_true = ttk.Label(imgs, text="TRUE (field_true.png)", anchor="center")
        self.lbl_pred = ttk.Label(imgs, text="PRED (field_pred.png)", anchor="center")
        self.lbl_err  = ttk.Label(imgs, text="ERR (field_err.png)", anchor="center")
        self.lbl_true.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.lbl_pred.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.lbl_err.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)

    # ----------------------------
    # Placement helpers
    # ----------------------------
    def _snap(self, x: float, y: float) -> Tuple[float, float]:
        if not self.snap_enabled.get():
            return x, y
        try:
            step = float(self.snap_step.get())
        except Exception:
            return x, y
        if step <= 0:
            return x, y
        return round(x / step) * step, round(y / step) * step

    def _canvas_to_world(self, px: float, py: float) -> Tuple[float, float]:
        w = max(1, self.preview.winfo_width())
        h = max(1, self.preview.winfo_height())
        pad = 14
        xmin, xmax = DOMAIN["xmin"], DOMAIN["xmax"]
        ymin, ymax = DOMAIN["ymin"], DOMAIN["ymax"]

        px = max(pad, min(w - pad, px))
        py = max(pad, min(h - pad, py))

        x = (px - pad) / (w - 2 * pad) * (xmax - xmin) + xmin
        y = ymax - (py - pad) / (h - 2 * pad) * (ymax - ymin)
        return self._snap(float(x), float(y))

    def _on_mouse_move(self, evt):
        x, y = self._canvas_to_world(evt.x, evt.y)
        self.cursor_var.set(f"X: {x:.3f}   Y: {y:.3f}")

    def _row_type(self, r: ObjRow) -> str:
        t = (r.type_cb.get() or "rect").strip().lower()
        return "circle" if t.startswith("c") else "rect"

    def _on_preview_click(self, evt):
        if not self.rows:
            return
        idx = max(1, min(len(self.rows), int(self.selected_obj_idx.get())))
        r = self.rows[idx - 1]

        x, y = self._canvas_to_world(evt.x, evt.y)
        mode = self.mode_var.get().strip()
        obj_type = self._row_type(r)

        def get_vals():
            try:
                return (
                    ffloat(r.x1_ent.get(), "x1"),
                    ffloat(r.x2_ent.get(), "x2"),
                    ffloat(r.y1_ent.get(), "y1"),
                    ffloat(r.y2_ent.get(), "y2"),
                )
            except Exception:
                return (0.5, 1.0, 0.5, 1.0)

        x1, x2, y1, y2 = get_vals()

        # RECT modes
        if mode.startswith("Rect"):
            r.type_cb.set("rect")

            if mode == "Rect: set corner 1":
                r.x1_ent.delete(0, "end"); r.x1_ent.insert(0, f"{x:.3f}")
                r.y1_ent.delete(0, "end"); r.y1_ent.insert(0, f"{y:.3f}")
                self._log(f"[Place] obj{idx} rect corner1=({x:.3f},{y:.3f})\n")

            elif mode == "Rect: set corner 2":
                r.x2_ent.delete(0, "end"); r.x2_ent.insert(0, f"{x:.3f}")
                r.y2_ent.delete(0, "end"); r.y2_ent.insert(0, f"{y:.3f}")
                self._log(f"[Place] obj{idx} rect corner2=({x:.3f},{y:.3f})\n")

            elif mode == "Rect: move center":
                cx_old = (x1 + x2) * 0.5
                cy_old = (y1 + y2) * 0.5
                dx = x - cx_old
                dy = y - cy_old
                r.x1_ent.delete(0, "end"); r.x1_ent.insert(0, f"{x1 + dx:.3f}")
                r.x2_ent.delete(0, "end"); r.x2_ent.insert(0, f"{x2 + dx:.3f}")
                r.y1_ent.delete(0, "end"); r.y1_ent.insert(0, f"{y1 + dy:.3f}")
                r.y2_ent.delete(0, "end"); r.y2_ent.insert(0, f"{y2 + dy:.3f}")
                self._log(f"[Move] obj{idx} rect center -> ({x:.3f},{y:.3f})\n")

        # CIRCLE modes
        else:
            r.type_cb.set("circle")

            # circle uses: x1=cx, y1=cy, x2=r
            cx = ffloat(r.x1_ent.get() or "1.0", "cx") if obj_type == "circle" else x
            cy = ffloat(r.y1_ent.get() or "1.0", "cy") if obj_type == "circle" else y
            rad = abs(ffloat(r.x2_ent.get() or "0.2", "r"))

            if mode == "Circle: set center":
                r.x1_ent.delete(0, "end"); r.x1_ent.insert(0, f"{x:.3f}")
                r.y1_ent.delete(0, "end"); r.y1_ent.insert(0, f"{y:.3f}")
                self._log(f"[Place] obj{idx} circle center=({x:.3f},{y:.3f})\n")

            elif mode == "Circle: set radius":
                # radius = distance from center to click
                try:
                    cx = ffloat(r.x1_ent.get(), "cx")
                    cy = ffloat(r.y1_ent.get(), "cy")
                except Exception:
                    cx, cy = x, y
                rad = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                r.x2_ent.delete(0, "end"); r.x2_ent.insert(0, f"{rad:.3f}")
                self._log(f"[Place] obj{idx} circle r={rad:.3f}\n")

        self.preview_scene()

    # ----------------------------
    # Object table
    # ----------------------------
    def apply_n(self):
        n = max(0, int(self.n_var.get()))
        cur = len(self.rows)

        if n > cur:
            for _ in range(n - cur):
                self._add_row()
        elif n < cur:
            for _ in range(cur - n):
                self._remove_row(self.rows[-1], renumber=False)

        self._renumber()
        self.sel_spin.configure(to=max(1, len(self.rows)))
        self.selected_obj_idx.set(min(self.selected_obj_idx.get(), max(1, len(self.rows))))
        self.preview_scene()
        self._log(f"Objects: {len(self.rows)}\n")

    def _add_row(self):
        fr = ttk.Frame(self.table_inner)
        fr.pack(fill="x", pady=2, padx=6)

        idx = ttk.Label(fr, text=str(len(self.rows) + 1), width=3, anchor="center")
        idx.grid(row=0, column=0, padx=3)

        type_cb = ttk.Combobox(fr, values=["rect", "circle"], state="readonly", width=6)
        type_cb.set("rect")
        type_cb.grid(row=0, column=1, padx=3)

        mat_cb = ttk.Combobox(fr, values=list(DEFAULT_MATERIALS.keys()), state="readonly", width=10)
        mat_cb.set("metal")
        mat_cb.grid(row=0, column=2, padx=3)

        x1 = ttk.Entry(fr, width=8); x1.insert(0, "0.50"); x1.grid(row=0, column=3, padx=3)
        x2 = ttk.Entry(fr, width=8); x2.insert(0, "1.00"); x2.grid(row=0, column=4, padx=3)
        y1 = ttk.Entry(fr, width=8); y1.insert(0, "0.50"); y1.grid(row=0, column=5, padx=3)
        y2 = ttk.Entry(fr, width=8); y2.insert(0, "1.00"); y2.grid(row=0, column=6, padx=3)

        btn = ttk.Button(fr, text="Delete", style="Danger.TButton")
        row = ObjRow(fr, idx, type_cb, mat_cb, x1, x2, y1, y2, btn)
        btn.configure(command=lambda r=row: self._remove_row(r))
        btn.grid(row=0, column=7, padx=3)

        self.rows.append(row)
        self.n_var.set(len(self.rows))

    def _remove_row(self, row: ObjRow, renumber=True):
        try:
            row.frame.destroy()
        except Exception:
            pass
        if row in self.rows:
            self.rows.remove(row)
        self.n_var.set(len(self.rows))
        if renumber:
            self._renumber()
        self.sel_spin.configure(to=max(1, len(self.rows)))
        self.selected_obj_idx.set(min(self.selected_obj_idx.get(), max(1, len(self.rows))))
        self.preview_scene()
        self._log(f"Deleted. Remaining: {len(self.rows)}\n")

    def _renumber(self):
        for i, r in enumerate(self.rows, start=1):
            r.idx_lbl.configure(text=str(i))

    def fill_examples(self):
        # rect, circle examples
        ex = [
            ("rect", "metal", 0.50, 1.00, 0.50, 1.00),
            ("circle", "glass", 1.80, 0.25, 1.00, 0.0),   # cx=1.8, r=0.25, cy=1.0
            ("rect", "brick", 2.10, 2.60, 1.20, 1.70),
        ]
        if not self.rows:
            self.apply_n()
        for i, r in enumerate(self.rows):
            t, mat, a, b, c, d = ex[i % len(ex)]
            r.type_cb.set(t)
            r.mat_cb.set(mat)
            if t == "rect":
                # a=x1 b=x2 c=y1 d=y2
                for ent, val in [(r.x1_ent, a), (r.x2_ent, b), (r.y1_ent, c), (r.y2_ent, d)]:
                    ent.delete(0, "end"); ent.insert(0, f"{val:.2f}")
            else:
                # a=cx b=r c=cy
                r.x1_ent.delete(0, "end"); r.x1_ent.insert(0, f"{a:.2f}")
                r.x2_ent.delete(0, "end"); r.x2_ent.insert(0, f"{b:.2f}")
                r.y1_ent.delete(0, "end"); r.y1_ent.insert(0, f"{c:.2f}")
                r.y2_ent.delete(0, "end"); r.y2_ent.insert(0, "0.00")
        self.preview_scene()
        self._log("Filled examples.\n")

    # ----------------------------
    # Scene build/save
    # ----------------------------
    def build_scene(self) -> dict:
        freq_ghz = ffloat(self.freq_var.get(), "Frequency (GHz)")
        if freq_ghz <= 0:
            raise ValueError("Frequency must be > 0")

        objects = []
        for i, r in enumerate(self.rows, start=1):
            mat = r.mat_cb.get().strip()
            obj_type = self._row_type(r)

            if obj_type == "rect":
                x1 = ffloat(r.x1_ent.get(), f"obj{i}.x1")
                x2 = ffloat(r.x2_ent.get(), f"obj{i}.x2")
                y1 = ffloat(r.y1_ent.get(), f"obj{i}.y1")
                y2 = ffloat(r.y2_ent.get(), f"obj{i}.y2")
                xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
                ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
                objects.append({"type": "rect", "material": mat, "x1": xa, "x2": xb, "y1": ya, "y2": yb})
            else:
                cx = ffloat(r.x1_ent.get(), f"obj{i}.cx")
                cy = ffloat(r.y1_ent.get(), f"obj{i}.cy")
                rad = abs(ffloat(r.x2_ent.get(), f"obj{i}.r"))
                objects.append({"type": "circle", "material": mat, "cx": cx, "cy": cy, "r": rad})

        return {
            "scene": {
                "source": {
                    "x0": SOURCE["x0"],
                    "y0": SOURCE["y0"],
                    "amplitude": SOURCE["amplitude"],
                    "frequency_hz": float(freq_ghz) * 1e9,
                },
                "air": {"absorption": DEFAULT_MATERIALS["air"]["absorption"]},
                "objects": objects,
            },
            "materials": DEFAULT_MATERIALS,
        }

    def save_scene(self) -> bool:
        try:
            data = self.build_scene()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

        with open(self.scene_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        self._log(f"Saved: {self.scene_path.name}\n")
        return True

    # ----------------------------
    # Preview
    # ----------------------------
    def _mat_color(self, mat: str) -> str:
        m = (mat or "air").strip().lower()
        if m in MATERIAL_COLORS:
            return MATERIAL_COLORS[m]
        # deterministic fallback
        idx = abs(hash(m)) % len(PALETTE)
        return PALETTE[idx]

    def preview_scene(self):
        self.preview.delete("all")
        w = max(1, self.preview.winfo_width())
        h = max(1, self.preview.winfo_height())
        pad = 14

        xmin, xmax = DOMAIN["xmin"], DOMAIN["xmax"]
        ymin, ymax = DOMAIN["ymin"], DOMAIN["ymax"]

        self.preview.create_rectangle(pad, pad, w - pad, h - pad, outline="#334155", width=2)

        def to_px(x, y):
            px = (x - xmin) / (xmax - xmin) * (w - 2 * pad) + pad
            py = (ymax - y) / (ymax - ymin) * (h - 2 * pad) + pad
            return px, py

        # source
        sx, sy = SOURCE["x0"], SOURCE["y0"]
        psx, psy = to_px(sx, sy)
        self.preview.create_oval(psx - 6, psy - 6, psx + 6, psy + 6, fill="#ef4444", outline="")
        self.preview.create_text(psx + 10, psy, fill="#e5e7eb", anchor="w", text="Source")

        # draw objects
        for i, r in enumerate(self.rows, start=1):
            mat = r.mat_cb.get().strip()
            col = self._mat_color(mat)
            t = self._row_type(r)

            try:
                if t == "rect":
                    x1 = ffloat(r.x1_ent.get(), "x1")
                    x2 = ffloat(r.x2_ent.get(), "x2")
                    y1 = ffloat(r.y1_ent.get(), "y1")
                    y2 = ffloat(r.y2_ent.get(), "y2")
                    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
                    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)

                    (xL, yB) = to_px(xa, ya)
                    (xR, yT) = to_px(xb, yb)

                    self.preview.create_rectangle(xL, yT, xR, yB, outline=col, width=2)
                    self.preview.create_rectangle(xL, yT, xR, yB, outline=col, width=0, fill=col, stipple="gray25")
                    self.preview.create_text(xL + 4, yT + 4, anchor="nw", fill="#e5e7eb", text=f"{i}:{mat}")
                else:
                    cx = ffloat(r.x1_ent.get(), "cx")
                    cy = ffloat(r.y1_ent.get(), "cy")
                    rad = abs(ffloat(r.x2_ent.get(), "r"))
                    (pcx, pcy) = to_px(cx, cy)
                    # approximate pixel radius using x-scale
                    pr = rad / (xmax - xmin) * (w - 2 * pad)

                    self.preview.create_oval(pcx - pr, pcy - pr, pcx + pr, pcy + pr, outline=col, width=2)
                    self.preview.create_oval(pcx - pr, pcy - pr, pcx + pr, pcy + pr, outline=col, width=0, fill=col, stipple="gray25")
                    self.preview.create_text(pcx + pr + 4, pcy - pr - 2, anchor="nw", fill="#e5e7eb", text=f"{i}:{mat}")
            except Exception:
                continue

        idx = int(self.selected_obj_idx.get())
        self.preview.create_text(
            pad + 6, pad + 6, anchor="nw", fill="#93c5fd",
            text=f"Selected: obj{idx} | Mode: {self.mode_var.get()} | Snap: {self.snap_enabled.get()} step={self.snap_step.get()}"
        )

    # ----------------------------
    # Results helpers
    # ----------------------------
    def _read_metrics(self, out_dir: Path) -> str:
        p = out_dir / "metrics.txt"
        if not p.exists():
            return "(metrics.txt not found)"
        return p.read_text(encoding="utf-8", errors="ignore").strip().replace("\n", " | ")

    def _set_image_label(self, label: ttk.Label, img_path: Path, key: str):
        if not img_path.exists():
            label.configure(text=f"Missing: {img_path.name}", image="")
            self._img_refs.pop(key, None)
            return
        target_w, target_h = 340, 240
        if PIL_OK:
            im = Image.open(img_path)
            im.thumbnail((target_w, target_h))
            tkimg = ImageTk.PhotoImage(im)
            label.configure(image=tkimg, text="")
            self._img_refs[key] = tkimg
        else:
            label.configure(text=f"Install pillow to display PNG\n{img_path.name}", image="")
            self._img_refs.pop(key, None)

    def show_results(self):
        out_dir = self.last_results_dir
        if not out_dir or not out_dir.exists():
            self.res_path_var.set("Results folder: (none)")
            self.res_metrics_var.set("Metrics: (none)")
            return

        self.res_path_var.set(f"Results folder: {out_dir}")
        self.res_metrics_var.set(f"Metrics: {self._read_metrics(out_dir)}")

        def pick(p1: Path, p2: Path) -> Path:
            return p1 if p1.exists() else p2

        self._set_image_label(self.lbl_true, pick(out_dir / "field_true.png", out_dir / "imgs" / "field_true.png"), "true")
        self._set_image_label(self.lbl_pred, pick(out_dir / "field_pred.png", out_dir / "imgs" / "field_pred.png"), "pred")
        self._set_image_label(self.lbl_err,  pick(out_dir / "field_err.png",  out_dir / "imgs" / "field_err.png"),  "err")

        self.nb.select(self.tab_res)

    # ----------------------------
    # Run solver (async)
    # ----------------------------
    def _start_poll(self):
        if self._polling:
            return
        self._polling = True
        self.after(50, self._poll_queue)

    def _stop_poll(self):
        self._polling = False

    def _try_progress_from_line(self, line: str):
        if "Epoch" in line and "/" in line:
            try:
                part = line.split("Epoch", 1)[1].strip().split()[0]  # "100/2000"
                a, b = part.split("/")
                e = float(a.strip())
                tot = float(b.strip())
                if tot > 0:
                    p = (e / tot) * 100.0
                    self.progress_var.set(max(0.0, min(100.0, p)))
                    self.progress_txt.set(f"{int(round(p))}%")
            except Exception:
                pass

    def _find_results_dir_from_line(self, line: str) -> Optional[Path]:
        m = re.search(r"Saved results:\s*(.+)$", line.strip())
        if m:
            p = Path(m.group(1).strip())
            if p.exists():
                return p
        return None

    def _fallback_latest_results_dir(self) -> Optional[Path]:
        base = self.project_dir / "simple_results"
        if not base.exists():
            return None
        dirs = [d for d in base.iterdir() if d.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda d: d.stat().st_mtime)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "line":
                    self._log(payload)
                    self._try_progress_from_line(payload)

                    found = self._find_results_dir_from_line(payload)
                    if found:
                        self.last_results_dir = found
                        self.btn_open.configure(state="normal")

                elif kind == "done":
                    self._finish_run_ui(payload)
        except queue.Empty:
            pass

        if self._polling:
            self.after(50, self._poll_queue)

    def run_solver(self):
        if self.running:
            messagebox.showinfo("Running", "Solver is already running.")
            return
        if not self.solver_path.exists():
            messagebox.showerror("Error", f"Solver not found:\n{self.solver_path}")
            return
        if not self.save_scene():
            return

        epochs = int(self.epochs_var.get())
        epochs = max(2000, min(200000, epochs))
        self.epochs_var.set(epochs)

        self.progress_var.set(0.0)
        self.progress_txt.set("0%")
        self.status_var.set("Running…")
        self.running = True
        self.btn_run.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.last_results_dir = None

        cmd = [sys.executable, str(self.solver_path), str(self.scene_path), "--epochs", str(epochs)]
        self._log(f"Running: {' '.join(cmd)}\n")
        self._start_poll()

        def worker():
            err = ""
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self._q.put(("line", line))
                self._proc.wait()
            except Exception as e:
                err = str(e)
            finally:
                self._q.put(("done", err))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_run_ui(self, err: str):
        self.running = False
        self.btn_run.configure(state="normal")
        self._stop_poll()
        self.progress_var.set(100.0)
        self.progress_txt.set("100%")
        self.status_var.set("Ready")

        if err:
            self._log(f"\nError: {err}\n")
            return

        if not self.last_results_dir or not self.last_results_dir.exists():
            self.last_results_dir = self._fallback_latest_results_dir()
            if self.last_results_dir and self.last_results_dir.exists():
                self.btn_open.configure(state="normal")

        if self.last_results_dir and self.last_results_dir.exists():
            self._log(f"Done. Results: {self.last_results_dir}\n")
            self.show_results()
        else:
            self._log("Done, but results folder not found.\n")

    def open_results(self):
        if self.last_results_dir and self.last_results_dir.exists():
            open_in_file_manager(self.last_results_dir)
        else:
            messagebox.showinfo("No results", "Run the solver first (▶ Run solver).")

    # ----------------------------
    # Restart
    # ----------------------------
    def restart_app(self):
        self.log.delete("1.0", "end")
        self._log("Restarted.\nReady: Apply N → Example/Preview → click placement → Save → Run\n")

        for w in self.table_inner.winfo_children():
            w.destroy()
        self.rows.clear()

        self.n_var.set(1)
        self.apply_n()
        self.fill_examples()

        self.last_results_dir = None
        self.btn_open.configure(state="disabled")
        self.res_path_var.set("Results folder: (none)")
        self.res_metrics_var.set("Metrics: (none)")
        self._img_refs.clear()
        self.lbl_true.configure(text="TRUE (field_true.png)", image="")
        self.lbl_pred.configure(text="PRED (field_pred.png)", image="")
        self.lbl_err.configure(text="ERR (field_err.png)", image="")
        self.nb.select(self.tab_log)

    # ----------------------------
    # Logging
    # ----------------------------
    def _log(self, s: str):
        self.log.insert("end", s)
        self.log.see("end")


if __name__ == "__main__":
    App().mainloop()
