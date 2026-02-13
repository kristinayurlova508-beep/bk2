import os
import sys
import re
import yaml
import queue
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import shutil

import tkinter as tk
from tkinter import ttk, messagebox

# Pillow нужен, чтобы показывать PNG в интерфейсе (вкладка Results)
try:
    from PIL import Image, ImageTk  # type: ignore
    PIL_OK = True
except Exception:
    PIL_OK = False

# ----------------------------
# MATERIALS (scene parameters)
# ----------------------------
# Default physical properties for each material used in the simulation.
# ----------------------------
# МАТЕРИАЛЫ (параметры для сцены)
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
# Colors used to draw points in the Preview canvas
# Цвета для отображения точек в Preview
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
# World boundaries (scene coordinate limits)
# Границы "мира" (координаты сцены)
WORLD = dict(xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)
# Default signal source parameters
# Параметры источника сигнала по умолчанию
DEFAULT_SOURCE = dict(x0=0.8, y0=1.0, amplitude=1.0, frequency_hz=1e9)


# ----------------------------
# Утилиты
# ----------------------------
def _to_float(s: str, default: float) -> float:
     
    """Безопасно парсим число из строки. Поддержка запятой как десятичного разделителя."""
    try:
        return float((s or "").strip().replace(",", "."))
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    """Ограничить значение диапазоном [lo, hi]."""
     #   """Clamp value into [lo, hi] range."""
    return max(lo, min(hi, v))


import shutil  # <-- должен быть в импортах

def open_path_default(path: Path):
    """Open file with default OS app (Windows/macOS/Linux/WSL)."""
    path = path.resolve()
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path)])
            return
        if "WSL_DISTRO_NAME" in os.environ:
            subprocess.Popen(["cmd.exe", "/c", "start", "", str(path)])
            return
        raise FileNotFoundError("No opener found (xdg-open/open/start).")
    except Exception as e:
        messagebox.showwarning("Cannot open", str(e))




def _wsl_to_windows_path(p: Path) -> str:
    """
    Convert /mnt/c/... path to C:\\... for Windows 'start'.
    Works only inside WSL when wslpath exists.
    """
    s = str(p.resolve())
    try:
        if "WSL_DISTRO_NAME" in os.environ and shutil.which("wslpath"):
            out = subprocess.check_output(["wslpath", "-w", s], text=True).strip()
            if out:
                return out
    except Exception:
        pass
    return s  # fallback

def open_path_default(path: Path):
    """Open file with default OS app (Windows/macOS/Linux/WSL-safe)."""
    path = path.resolve()
    try:
        # WSL: open via Windows (best), convert path first
        if "WSL_DISTRO_NAME" in os.environ:
            win_path = _wsl_to_windows_path(path)
            subprocess.Popen(["cmd.exe", "/c", "start", "", win_path])
            return

        # Native Windows
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return

        # macOS
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return

        # Linux (only if xdg-open exists)
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path)])
            return

        raise FileNotFoundError("No opener found (WSL/Windows/macOS/xdg-open).")

    except Exception as e:
        messagebox.showwarning("Cannot open", str(e))

def open_folder_default(folder: Path):
    """Open folder in file explorer (Windows/macOS/Linux/WSL-safe)."""
    folder = folder.resolve()
    try:
        # WSL: open via Windows Explorer, convert path
        if "WSL_DISTRO_NAME" in os.environ:
            win_path = _wsl_to_windows_path(folder)
            subprocess.Popen(["cmd.exe", "/c", "start", "", win_path])
            return

        # Native Windows
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
            return

        # macOS
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
            return

        # Linux
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(folder)])
            return

        raise FileNotFoundError("No opener found (WSL/Windows/macOS/xdg-open).")

    except Exception as e:
        messagebox.showwarning("Cannot open folder", str(e))

# ---- back-compat aliases (если у тебя где-то старые имена) ----
def _open_file_default(path: Path):
    open_path_default(path)

def _open_folder(path: Path):
    open_folder_default(path)


def _wsl_to_windows_path(p: Path) -> str:
    s = str(p.resolve())
    try:
        if "WSL_DISTRO_NAME" in os.environ and shutil.which("wslpath"):
            return subprocess.check_output(["wslpath", "-w", s], text=True).strip()
    except Exception:
        pass
    return s

def open_folder_default(folder: Path):
    folder = folder.resolve()
    try:
        # WSL -> открыть через Windows Explorer
        if "WSL_DISTRO_NAME" in os.environ:
            win_path = _wsl_to_windows_path(folder)
            subprocess.Popen(["cmd.exe", "/c", "start", "", win_path])
            return

        # Windows
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
            return

        # macOS
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
            return

        # Linux
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(folder)])
            return

        raise FileNotFoundError("No folder opener found.")
    except Exception as e:
        messagebox.showwarning("Cannot open folder", str(e))

# алиас, если в коде где-то используется старое имя
def _open_folder(path: Path):
    open_folder_default(path)


# Строка таблицы объектов (для удобства хранения виджетов)
@dataclass
class PointRow:
    frame: ttk.Frame
    idx_lbl: ttk.Label
    mat_cb: ttk.Combobox
    x_ent: ttk.Entry
    y_ent: ttk.Entry
    del_btn: ttk.Button

#"""Open file with the system default application (Windows/macOS/Linux/WSL)."""
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scene Builder — Points")
        self.geometry("1400x900")
        self.minsize(1180, 720)

        # Папка проекта и пути к файлам Project folder and file paths
        self.project_dir = Path(__file__).resolve().parent
        self.scene_path = self.project_dir / "scene.yaml"
        self.solver_path = self.project_dir / "solver_simple_torch.py"

        # Список объектов сцены (каждый объект: x, y, material)
        # List of scene objects (each object: x, y, material)
        self.objects: List[dict] = []

        # GUI-переменные (связаны с виджетами)
        # GUI variables (related to widgets)
        self.n_var = tk.IntVar(value=5)
        self.epochs_var = tk.IntVar(value=2000)
        self.freq_ghz_var = tk.StringVar(value="1.0")
        self.selected_obj_idx = tk.IntVar(value=1)

        
        # Binding to the grid (snap)
        self.snap_enabled = tk.BooleanVar(value=True)
        self.snap_step = tk.StringVar(value="0.05")

        # Статусы # Statuses
        self.cursor_var = tk.StringVar(value="X: -, Y: -")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_txt = tk.StringVar(value="0%")

        # Строки таблицы # Rows of the table
        self.rows: List[PointRow] = []

        # Для запуска solver в отдельном потоке и чтения логов
        self.running = False
        self.last_results_dir: Optional[Path] = None
        self._q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._polling = False

        # Нужно хранить ссылки на изображения Tk, иначе они "пропадут" (GC)
        self._img_refs = {}

        self._style()
        self._ui()

        # Инициализация интерфейса данными
        self.apply_n()
        self.fill_example()

        # Чуть позже отрисовать превью
        self.after(80, self.preview_scene)

    # ----------------------------
    # Стили интерфейса
    # ----------------------------
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

    # ----------------------------
    # Построение UI
    # ----------------------------
    def _ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Верхняя панель
        head = ttk.Frame(root)
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="Scene Builder (Points only)", style="Header.TLabel").pack(side="left")

        # Статус справа + прогресс-бар
        right_head = ttk.Frame(head)
        right_head.pack(side="right")
        ttk.Label(right_head, textvariable=self.status_var).pack(side="right", padx=(0, 10))
        ttk.Label(right_head, textvariable=self.progress_txt).pack(side="right")
        ttk.Progressbar(
            right_head, orient="horizontal", length=260,
            mode="determinate", maximum=100.0, variable=self.progress_var
        ).pack(side="right", padx=(8, 8))

        # Controls
        ctrl = ttk.LabelFrame(root, text="Controls")
        ctrl.pack(fill="x", pady=(0, 10))

        bar = ttk.Frame(ctrl)
        bar.pack(fill="x", padx=8, pady=8)

        ttk.Label(bar, text="Frequency (GHz):").pack(side="left")
        ttk.Entry(bar, textvariable=self.freq_ghz_var, width=8).pack(side="left", padx=(6, 14))

        ttk.Label(bar, text="Objects (N):").pack(side="left")
        ttk.Spinbox(bar, from_=0, to=200, textvariable=self.n_var, width=6).pack(side="left", padx=(6, 8))
        ttk.Button(bar, text="Apply N", command=self.apply_n).pack(side="left", padx=(0, 14))

        ttk.Label(bar, text="Epochs:").pack(side="left")
        ttk.Spinbox(bar, from_=100, to=20000, increment=100, textvariable=self.epochs_var, width=9)\
            .pack(side="left", padx=(6, 14))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(bar, text="Example", command=self.fill_example).pack(side="left", padx=4)
        ttk.Button(bar, text="Preview", command=self.preview_scene).pack(side="left", padx=4)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(bar, text="Save scene.yaml", command=self.save_scene).pack(side="left", padx=4)
        self.btn_run = ttk.Button(bar, text="▶ Run solver", style="Primary.TButton", command=self.run_solver)
        self.btn_run.pack(side="left", padx=6)

        self.btn_open = ttk.Button(bar, text="Open results folder", command=self.open_results, state="disabled")
        self.btn_open.pack(side="left", padx=4)

        self.btn_restart = ttk.Button(bar, text="Restart", command=self.restart_app)
        self.btn_restart.pack(side="right", padx=4)
    # Path to scene.yaml file
        # Путь к scene.yaml
        self.path_var = tk.StringVar(value=f"scene.yaml: {self.scene_path}")
        ttk.Label(ctrl, textvariable=self.path_var).pack(anchor="w", padx=10, pady=(0, 8))
 # Placement controls – select object + snap to grid
        # Placement - выбор объекта + snap
        place = ttk.LabelFrame(root, text="Placement (click on Preview to move selected point)")
        place.pack(fill="x", pady=(0, 10))
        pl = ttk.Frame(place)
        pl.pack(fill="x", padx=8, pady=8)

        ttk.Label(pl, text="Selected object:").pack(side="left")
        self.sel_spin = ttk.Spinbox(pl, from_=1, to=1, textvariable=self.selected_obj_idx, width=6)
        self.sel_spin.pack(side="left", padx=(6, 16))

        ttk.Separator(pl, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Checkbutton(pl, text="Snap", variable=self.snap_enabled).pack(side="left")
        ttk.Label(pl, text="Step:").pack(side="left", padx=(8, 4))
        ttk.Combobox(pl, values=["0.01", "0.05", "0.1"], state="readonly",
                     textvariable=self.snap_step, width=6).pack(side="left", padx=(0, 12))
  # Main layout area (left: table + preview, right: tabs)
        # Основная область (слева таблица+превью, справа вкладки)
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=2)
        main.rowconfigure(1, weight=3)
  # Objects table container
        # Таблица объектов
        obj_box = ttk.LabelFrame(main, text="Objects (points)")
        obj_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))

        hdr = ttk.Frame(obj_box)
        hdr.pack(fill="x", pady=(0, 6), padx=6)
        ttk.Label(hdr, text="#", width=4, anchor="center").grid(row=0, column=0, padx=3)
        ttk.Label(hdr, text="Material", width=14, anchor="center").grid(row=0, column=1, padx=3)
        ttk.Label(hdr, text="X", width=10, anchor="center").grid(row=0, column=2, padx=3)
        ttk.Label(hdr, text="Y", width=10, anchor="center").grid(row=0, column=3, padx=3)
        ttk.Label(hdr, text="", width=10, anchor="center").grid(row=0, column=4, padx=3)
      # Scrollable table: Canvas + inner Frame
        # Прокручиваемая таблица: Canvas + Frame внутри
        self.table_canvas = tk.Canvas(obj_box, highlightthickness=0)
        self.table_scroll = ttk.Scrollbar(obj_box, orient="vertical", command=self.table_canvas.yview)
        self.table_inner = ttk.Frame(self.table_canvas)
        self.table_inner.bind("<Configure>", lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all")))
        self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")
        self.table_canvas.configure(yscrollcommand=self.table_scroll.set)
        self.table_canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        self.table_scroll.pack(side="right", fill="y", padx=(0, 6), pady=(0, 6))
# Preview canvas container
        # Preview холст
        prev_box = ttk.LabelFrame(main, text="Preview")
        prev_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.preview = tk.Canvas(prev_box, background="#0b1220")
        self.preview.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        self.preview.bind("<Button-1>", self._on_preview_click)     # клик: перемещение выбранной точки
        self.preview.bind("<Motion>", self._on_mouse_move)          # движение мыши: вывод координат
        self.preview.bind("<Configure>", lambda e: self.preview_scene())  # изменение размера: перерисовка
        ttk.Label(prev_box, textvariable=self.cursor_var, anchor="w").pack(fill="x", padx=10, pady=(4, 6))
# Mouse move – show world coordinates
        # Правая часть: Notebook (Log/Results)
        right = ttk.Notebook(main)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
# Log tab
        # Лог
        log_tab = ttk.Frame(right)
        right.add(log_tab, text="Log")
        self.log = tk.Text(log_tab, height=10, wrap="word", background="#0b1220", foreground="#e5e7eb")
        self.log.pack(fill="both", expand=True, padx=6, pady=6)
 # Results tab
        # Результаты
        res_tab = ttk.Frame(right)
        right.add(res_tab, text="Results")
 # Results tab
        # ✅ Results: горизонтальная и вертикальная прокрутка
        self.res_canvas = tk.Canvas(res_tab, highlightthickness=0)
        self.res_hscroll = ttk.Scrollbar(res_tab, orient="horizontal", command=self.res_canvas.xview)
        self.res_vscroll = ttk.Scrollbar(res_tab, orient="vertical", command=self.res_canvas.yview)
        self.res_canvas.configure(xscrollcommand=self.res_hscroll.set, yscrollcommand=self.res_vscroll.set)

        self.res_canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        self.res_vscroll.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self.res_hscroll.pack(side="bottom", fill="x", padx=6, pady=(0, 6))

        self.res_inner = ttk.Frame(self.res_canvas)
        self.res_inner.bind("<Configure>", lambda e: self.res_canvas.configure(scrollregion=self.res_canvas.bbox("all")))
        self.res_canvas.create_window((0, 0), window=self.res_inner, anchor="nw")

        if not PIL_OK:
            ttk.Label(self.res_inner, text="Install pillow to show images: pip install pillow").grid(row=0, column=0, sticky="w", padx=8, pady=8)

        # ----------------------------
    # Objects (points)
    # Объекты (точки)
    # ----------------------------
    def apply_n(self):
        """
        Apply N (number of objects). Adds/removes points and updates table/preview.
        Применить N (кол-во объектов). Добавляет/удаляет точки и обновляет таблицу/превью.
        """
        n = int(self.n_var.get())
        if n < 0:
            n = 0

        # Configure selected object spinbox range
        # Настроить диапазон спинбокса выбранного объекта
        self.sel_spin.configure(from_=1, to=max(1, n))
        if self.selected_obj_idx.get() > max(1, n):
            self.selected_obj_idx.set(max(1, n))

        # Resize objects list to match N
        # Догнать список объектов до нужного размера
        while len(self.objects) < n:
            self.objects.append({"x": 1.0, "y": 1.0, "material": "concrete"})
        while len(self.objects) > n:
            self.objects.pop()

        self._rebuild_table()
        self.preview_scene()

    def _rebuild_table(self):
        """
        Rebuild the entire objects table (simple and reliable approach).
        Полностью пересобирает таблицу объектов (простое и надежное решение).
        """
        # Clear table UI
        # Очищаем таблицу UI
        for w in self.table_inner.winfo_children():
            w.destroy()
        self.rows.clear()

        # Available materials list (sorted)
        # Список доступных материалов (отсортирован)
        mats = sorted(DEFAULT_MATERIALS.keys())

        for i, obj in enumerate(self.objects, start=1):
            fr = ttk.Frame(self.table_inner)
            fr.pack(fill="x", padx=6, pady=2)

            ttk.Label(fr, text=str(i), width=4, anchor="center").grid(row=0, column=0, padx=3)

            # Material dropdown
            # Выбор материала
            mat_cb = ttk.Combobox(fr, values=mats, state="readonly", width=14)
            mat_cb.set(obj.get("material", "concrete"))
            mat_cb.grid(row=0, column=1, padx=3)

            # Coordinate inputs
            # Поля координат
            x_ent = ttk.Entry(fr, width=10)
            y_ent = ttk.Entry(fr, width=10)
            x_ent.grid(row=0, column=2, padx=3)
            y_ent.grid(row=0, column=3, padx=3)
            x_ent.insert(0, f'{float(obj["x"]):.3f}')
            y_ent.insert(0, f'{float(obj["y"]):.3f}')

            # Delete button (remove object)
            # Кнопка удаления объекта
            del_btn = ttk.Button(
                fr, text="Delete", style="Danger.TButton",
                command=lambda ii=i: self.delete_object(ii)
            )
            del_btn.grid(row=0, column=4, padx=3)

            # Callback: material changed
            # Коллбек: смена материала
            def make_mat_cb(ii: int, cb: ttk.Combobox):
                def _on(_evt=None):
                    self.objects[ii]["material"] = (cb.get() or "concrete").strip().lower()
                    self.preview_scene()
                return _on

            # Callback: apply X/Y changes (Enter or focus out)
            # Коллбек: применить X/Y (Enter или потеря фокуса)
            def make_xy_apply(ii: int, ex: ttk.Entry, ey: ttk.Entry):
                def _apply(_evt=None):
                    x = _to_float(ex.get(), float(self.objects[ii]["x"]))
                    y = _to_float(ey.get(), float(self.objects[ii]["y"]))

                    # Clamp to world boundaries
                    # Ограничиваем границами мира
                    x = _clamp(x, WORLD["xmin"], WORLD["xmax"])
                    y = _clamp(y, WORLD["ymin"], WORLD["ymax"])

                    self.objects[ii]["x"] = float(x)
                    self.objects[ii]["y"] = float(y)

                    # Write normalized formatting back to entries
                    # Записываем форматированные значения обратно в поля
                    ex.delete(0, "end"); ex.insert(0, f"{x:.3f}")
                    ey.delete(0, "end"); ey.insert(0, f"{y:.3f}")

                    self.preview_scene()
                return _apply

            mat_cb.bind("<<ComboboxSelected>>", make_mat_cb(i - 1, mat_cb))
            apply_xy = make_xy_apply(i - 1, x_ent, y_ent)
            x_ent.bind("<Return>", apply_xy)
            y_ent.bind("<Return>", apply_xy)
            x_ent.bind("<FocusOut>", apply_xy)
            y_ent.bind("<FocusOut>", apply_xy)

        # Update selected object spinbox range after rebuild
        # Настроить диапазон выбранного объекта после пересборки
        n = len(self.objects)
        self.sel_spin.configure(from_=1, to=max(1, n))
        if n == 0:
            self.selected_obj_idx.set(1)
        else:
            self.selected_obj_idx.set(_clamp(int(self.selected_obj_idx.get()), 1, n))

    def delete_object(self, idx_1based: int):
        """
        Delete an object by index (1-based).
        Удалить объект по номеру (начиная с 1).
        """
        if idx_1based < 1 or idx_1based > len(self.objects):
            return
        self.objects.pop(idx_1based - 1)
        self.n_var.set(len(self.objects))
        self._rebuild_table()
        self.preview_scene()

    def fill_example(self):
        """
        Fill scene with example points (different materials).
        Заполнить объекты примером (несколько точек разных материалов).
        """
        n = max(5, int(self.n_var.get()))
        self.n_var.set(n)
        self.apply_n()

        base = [
            (0.7, 0.6, "concrete"),
            (1.2, 1.3, "metal"),
            (1.8, 0.9, "glass"),
            (2.3, 1.6, "plastic"),
            (2.7, 0.4, "water"),
        ]
        for i in range(n):
            x, y, m = base[i % len(base)]
            self.objects[i]["x"] = x
            self.objects[i]["y"] = y
            self.objects[i]["material"] = m

        self._rebuild_table()
        self.preview_scene()
        self._log("[Example] Loaded points.\n")

    # ----------------------------
    # Coordinate transforms + Preview rendering
    # Математика преобразования координат + отрисовка Preview
    # ----------------------------
    def _world_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert world coordinates (x,y) to canvas pixel coordinates (px,py).
        Перевод координат мира (x,y) в координаты Canvas (px,py).
        """
        w = self.preview.winfo_width() or 1
        h = self.preview.winfo_height() or 1
        xmin, xmax, ymin, ymax = WORLD["xmin"], WORLD["xmax"], WORLD["ymin"], WORLD["ymax"]
        px = (x - xmin) / (xmax - xmin) * w
        py = h - (y - ymin) / (ymax - ymin) * h
        return px, py

    def _canvas_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """
        Convert canvas pixel coordinates (px,py) back to world coordinates (x,y).
        Перевод координат Canvas (px,py) обратно в координаты мира (x,y).
        """
        w = self.preview.winfo_width() or 1
        h = self.preview.winfo_height() or 1
        xmin, xmax, ymin, ymax = WORLD["xmin"], WORLD["xmax"], WORLD["ymin"], WORLD["ymax"]
        x = xmin + (px / w) * (xmax - xmin)
        y = ymin + ((h - py) / h) * (ymax - ymin)
        return x, y

    def _snap(self, v: float) -> float:
        """
        Snap coordinate to nearest grid step (if enabled).
        Если включен Snap — округлить координату к ближайшему шагу.
        """
        if not self.snap_enabled.get():
            return v
        try:
            step = float(self.snap_step.get())
            if step <= 0:
                return v
        except Exception:
            return v
        return round(v / step) * step

    def preview_scene(self):
        """
        Fully redraw the Preview canvas.
        Полная перерисовка холста Preview.
        """
        self.preview.delete("all")
        w = self.preview.winfo_width() or 1
        h = self.preview.winfo_height() or 1

        # Draw background grid (visual aid only)
        # Рисуем сетку (только для визуального удобства)
        for i in range(1, 6):
            x = i * w / 6
            self.preview.create_line(x, 0, x, h, fill="#111827")
        for i in range(1, 4):
            y = i * h / 4
            self.preview.create_line(0, y, w, y, fill="#111827")

        # Determine selected object index
        # Индекс выбранного объекта
        sel = -1
        if self.objects:
            sel = max(1, min(len(self.objects), int(self.selected_obj_idx.get()))) - 1

        # Draw all points
        # Рисуем точки
        for i, obj in enumerate(self.objects):
            col = MATERIAL_COLORS.get(obj.get("material", "concrete"), "#22c55e")
            px, py = self._world_to_canvas(float(obj["x"]), float(obj["y"]))
            r = 6
            self.preview.create_oval(px - r, py - r, px + r, py + r, fill=col, outline="#0b1220")
            self.preview.create_text(px + 10, py, text=str(i + 1), fill="#e5e7eb", anchor="w")

            # Highlight selected point with an outer ring
            # Выделяем выбранный объект кружком
            if i == sel:
                self.preview.create_oval(
                    px - (r + 3), py - (r + 3), px + (r + 3), py + (r + 3),
                    outline="#ffffff", width=2
                )

    def _on_mouse_move(self, evt):
        """
        Show world coordinates under mouse cursor.
        При движении мыши по Preview показываем координаты мира.
        """
        x, y = self._canvas_to_world(evt.x, evt.y)
        self.cursor_var.set(f"X: {x:.3f}, Y: {y:.3f}")

    def _on_preview_click(self, evt):
        """
        Move selected point to click position (with snap if enabled).
        Клик по Preview: перемещаем выбранную точку в место клика (с учетом Snap).
        """
        if not self.objects:
            return

        sel = max(1, min(len(self.objects), int(self.selected_obj_idx.get()))) - 1
        x, y = self._canvas_to_world(evt.x, evt.y)
        x = self._snap(x)
        y = self._snap(y)

        self.objects[sel]["x"] = float(x)
        self.objects[sel]["y"] = float(y)

        # Quick way to refresh table values: rebuild table from scratch
        # Быстро обновить X/Y в таблице: пересобрать таблицу целиком
        self._rebuild_table()
        self.preview_scene()

    # ----------------------------
    # YAML saving
    # Сохранение YAML
    # ----------------------------
    def build_scene_dict(self) -> dict:
        """
        Build data structure for scene.yaml.
        Собрать структуру данных для scene.yaml.
        """
        freq_ghz = _to_float(self.freq_ghz_var.get(), 1.0)

        # Copy default source and set frequency in Hz
        # Копируем источник по умолчанию и задаём частоту в Гц
        src = dict(DEFAULT_SOURCE)
        src["frequency_hz"] = float(freq_ghz) * 1e9

        # Serialize objects list
        # Сериализуем список объектов
        objs = []
        for o in self.objects:
            objs.append({
                "x": float(o["x"]),
                "y": float(o["y"]),
                "material": str(o.get("material", "concrete")).strip().lower(),
            })

        return {
            "materials": DEFAULT_MATERIALS,
            "scene": {
                "source": src,
                "objects": objs,
            }
        }

    def save_scene(self):
        """
        Write scene.yaml to disk.
        Записать scene.yaml.
        """
        try:
            data = self.build_scene_dict()
            with open(self.scene_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

            self._log(f"[Save] Wrote {self.scene_path}\n")
            self.path_var.set(f"scene.yaml: {self.scene_path}")
        except Exception as e:
            messagebox.showerror("Cannot save", str(e))

    # ----------------------------
    # Run solver in background thread
    # Запуск solver в отдельном потоке
    # ----------------------------
    def run_solver(self):
        """
        Run solver_simple_torch.py and parse stdout for progress updates.
        Запуск solver_simple_torch.py и чтение stdout для прогресса.
        """
        if self.running:
            return
        if not self.solver_path.exists():
            messagebox.showerror("Missing solver", f"Cannot find solver: {self.solver_path}")
            return

        self.save_scene()

        # Clamp epochs and round down to hundreds
        # Ограничиваем epochs и округляем вниз до сотен
        epochs = int(self.epochs_var.get())
        epochs = max(100, min(20000, epochs))
        epochs = (epochs // 100) * 100
        self.epochs_var.set(epochs)

        cmd = [sys.executable, str(self.solver_path), str(self.scene_path), "--epochs", str(epochs)]
        self._log(f"[Run] {' '.join(cmd)}\n")

        # Update UI state for running process
        # Обновляем UI состояние на время выполнения
        self.running = True
        self.status_var.set("Running...")
        self.progress_var.set(0.0)
        self.progress_txt.set("0%")
        self.btn_run.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.last_results_dir = None

        def worker():
            """
            Worker thread: start process and read stdout line by line.
            Поток-воркер: запускаем процесс и читаем stdout построчно.
            """
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(self.project_dir),
                    bufsize=1,
                    universal_newlines=True,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._q.put(("log", line))
                rc = proc.wait()
                self._q.put(("done", str(rc)))
            except Exception as e:
                self._q.put(("log", f"[Run] Error: {e}\n"))
                self._q.put(("done", "1"))

        threading.Thread(target=worker, daemon=True).start()

        # Start polling queue events in UI thread
        # Запускаем периодический опрос очереди в UI потоке
        if not self._polling:
            self._polling = True
            self.after(80, self._poll_queue)

    def _poll_queue(self):
        """
        UI timer: drain queue events and update UI.
        UI-таймер: забирает события из очереди и обновляет интерфейс.
        """
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._handle_solver_line(payload)
                elif kind == "done":
                    self._on_solver_done(int(payload))
        except queue.Empty:
            pass

        if self.running:
            self.after(80, self._poll_queue)
        else:
            self._polling = False

    def _handle_solver_line(self, line: str):
        """
        Parse solver log line: progress and results directory path.
        Парсим строки лога solver: прогресс и путь к результатам.
        """
        self._log(line)

        # Progress parsing: expected format like "Epoch 100 / 2000"
        # Прогресс: ищем строку вида "Epoch 100 / 2000"
        m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            e = int(m.group(1))
            E = max(1, int(m.group(2)))
            pct = max(0.0, min(100.0, 100.0 * (e / E)))
            self.progress_var.set(pct)
            self.progress_txt.set(f"{pct:.0f}%")

        # Results folder detection: expected format "Saved results: <path>"
        # Папка результатов: ищем строку "Saved results: <path>"
        m2 = re.search(r"Saved results:\s*(.+)", line)
        if m2:
            p = m2.group(1).strip()
            self.last_results_dir = Path(p)
            self.btn_open.configure(state="normal")
            self._load_results_images()

    def _on_solver_done(self, rc: int):
        """
        Called when solver exits; restore UI state.
        Когда solver завершился — вернуть UI в нормальное состояние.
        """
        self.running = False
        self.btn_run.configure(state="normal")
        self.status_var.set("Ready" if rc == 0 else f"Error (code {rc})")
        if rc == 0:
            self.progress_var.set(100.0)
            self.progress_txt.set("100%")

    def open_results(self):
        """
        Open results directory.
        Открыть папку результатов.
        """
        if not self.last_results_dir:
            return
        if self.last_results_dir.exists():
            self._open_path_default(self.last_results_dir)
        else:
            messagebox.showwarning("Missing", f"Folder not found:\n{self.last_results_dir}")

    def _open_path_default(self, path: Path):
        """
        Open folder with OS file explorer.
        Открыть папку средствами ОС.
        """
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path.resolve()))  # type: ignore[attr-defined]
                return
            subprocess.Popen(["xdg-open", str(path.resolve())])
        except Exception as e:
            messagebox.showwarning("Cannot open folder", str(e))

    # ----------------------------
    # Results tab: show 3 images (Pred/True/Err)
    # Results: показываем 3 картинки в ряд (Pred/True/Err)
    # ----------------------------
    def _load_results_images(self):
        """
        Load field_pred.png / field_true.png / field_err.png into Results tab.
        Загрузить field_pred.png / field_true.png / field_err.png в вкладку Results.
        """
        if not PIL_OK or not self.last_results_dir:
            return

        # Clear previous content
        # Очищаем вкладку
        for w in self.res_inner.winfo_children():
            w.destroy()

        imgs = [
            ("Pred", self.last_results_dir / "field_pred.png"),
            ("True", self.last_results_dir / "field_true.png"),
            ("Err", self.last_results_dir / "field_err.png"),
        ]

        for col_i, (title, path) in enumerate(imgs):
            card = ttk.LabelFrame(self.res_inner, text=title)
            card.grid(row=0, column=col_i, padx=10, pady=10, sticky="n")

            if not path.exists():
                ttk.Label(card, text=f"Missing: {path.name}").pack(anchor="w", padx=8, pady=8)
                continue

            try:
                im = Image.open(path)

                # Scale by height so all images have the same preview height
                # Масштабируем по высоте, чтобы все были одинаковые по размеру
                target_h = 260
                if im.height > 0:
                    scale = target_h / im.height
                    im = im.resize((int(im.width * scale), int(im.height * scale)))

                tk_im = ImageTk.PhotoImage(im)

                # Keep reference to avoid image being garbage-collected
                # Сохраняем ссылку, иначе картинка исчезнет
                self._img_refs[title] = tk_im

                lbl = ttk.Label(card, image=tk_im)
                lbl.pack(padx=8, pady=8)

                # Open image file externally
                # Открыть файл изображения внешним приложением
                ttk.Button(card, text="Open", command=lambda p=path: _open_file_default(p)).pack(pady=(0, 8))

            except Exception as e:
                ttk.Label(card, text=f"Cannot load: {e}").pack(anchor="w", padx=8, pady=8)

        # Make columns stretch nicely
        # Растягиваем колонки равномерно
        self.res_inner.grid_columnconfigure(0, weight=1)
        self.res_inner.grid_columnconfigure(1, weight=1)
        self.res_inner.grid_columnconfigure(2, weight=1)

    # ----------------------------
    # Misc
    # Разное
    # ----------------------------
    def _log(self, s: str):
        """
        Append text to log widget.
        Добавить текст в окно лога.
        """
        self.log.insert("end", s)
        self.log.see("end")

    def restart_app(self):
        """
        Restart application (execv re-launches the current python process).
        Перезапуск приложения (execv запускает заново текущий python-процесс).
        """
        try:
            python = sys.executable
            os.execv(python, [python] + sys.argv)
        except Exception as e:
            messagebox.showerror("Cannot restart", str(e))


# Script entry point
# Точка входа
if __name__ == "__main__":
    App().mainloop()
