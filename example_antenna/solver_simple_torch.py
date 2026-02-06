import argparse
import math
import time
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# optional: VTK export
try:
    import pyvista as pv
    PV_OK = True
except Exception:
    PV_OK = False


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


def load_scene(scene_path: Path) -> dict:
    data = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    if "scene" not in data:
        raise ValueError("scene.yaml missing 'scene' key")
    if "objects" not in data["scene"]:
        data["scene"]["objects"] = []
    if "materials" not in data or not isinstance(data["materials"], dict):
        data["materials"] = {}
    # merge defaults for safety
    merged = dict(DEFAULT_MATERIALS)
    merged.update(data["materials"])
    data["materials"] = merged
    return data


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def k_from_freq(freq_hz: float) -> float:
    c = 3e8
    return 2.0 * math.pi * (freq_hz / c)


def mat_params(mat: str, materials: dict) -> tuple[float, float]:
    m = materials.get(mat, None)
    if m is None:
        # unknown material -> fallback to "air"
        m = materials.get("air", {"absorption": 0.05, "scatter": 0.0})
    absorption = float(m.get("absorption", 0.5))
    scatter = float(m.get("scatter", 0.2))
    return absorption, scatter


def make_grid(nx=360, ny=240, xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, X, Y


def field_true_numpy(scene: dict, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    src = scene["scene"]["source"]
    mats = scene.get("materials", {})

    x0 = float(src.get("x0", 0.8))
    y0 = float(src.get("y0", 1.0))
    amp = float(src.get("amplitude", 1.0))
    f = float(src.get("frequency_hz", 1e9))
    k = k_from_freq(f)

    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) + 1e-6
    base = amp * np.sin(k * r) / np.sqrt(r)

    obj_list = scene["scene"].get("objects", [])
    scat = 0.0

    for o in obj_list:
        t = str(o.get("type", "rect")).lower()
        mat = str(o.get("material", "air"))
        absorption, scatter_gain = mat_params(mat, mats)

        if t == "circle":
            cx = float(o["cx"]); cy = float(o["cy"]); rr = float(o["r"])
            mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= (rr ** 2)
            base = base.copy()
            base[mask] = base[mask] * np.exp(-0.08 * absorption)

            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + 1e-6
            scat += (0.30 * scatter_gain) * np.sin(k * dist + 0.4 * scatter_gain) / np.sqrt(dist)
        else:
            x1 = float(o["x1"]); x2 = float(o["x2"])
            y1 = float(o["y1"]); y2 = float(o["y2"])
            xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
            ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
            mask = (X >= xa) & (X <= xb) & (Y >= ya) & (Y <= yb)

            base = base.copy()
            base[mask] = base[mask] * np.exp(-0.08 * absorption)

            cx = 0.5 * (xa + xb)
            cy = 0.5 * (ya + yb)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + 1e-6
            scat += (0.30 * scatter_gain) * np.sin(k * dist + 0.4 * scatter_gain) / np.sqrt(dist)

    return (base + scat).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim=2, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def export_vti(out_path: Path, U: np.ndarray, xs: np.ndarray, ys: np.ndarray, name: str):
    if not PV_OK:
        return
    ny, nx = U.shape
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, 1)
    grid.origin = (float(xs[0]), float(ys[0]), 0.0)
    grid.spacing = (dx, dy, 1.0)

    grid.point_data[name] = U.reshape(-1, order="F")
    grid.save(str(out_path))


def save_objects_vtm(out_dir: Path, scene_data: dict):
    if not PV_OK:
        print("[WARN] pyvista/vtk not installed -> skip objects.vtm")
        return

    blocks = pv.MultiBlock()
    objs = scene_data.get("scene", {}).get("objects", []) or []

    for i, o in enumerate(objs, start=1):
        t = str(o.get("type", "rect")).lower()
        mat = str(o.get("material", "air"))

        if t == "circle":
            cx = float(o["cx"]); cy = float(o["cy"]); r = float(o["r"])
            n = 64
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.zeros_like(ang)], axis=1).astype(float)
            faces = np.concatenate([[n], np.arange(n, dtype=np.int64)])
            poly = pv.PolyData(pts, faces)
            poly["id"] = np.full(poly.n_points, i, dtype=np.int32)
            poly["material"] = np.array([mat] * poly.n_points, dtype=object)
            blocks.append(poly)
        else:
            x1, x2 = float(o["x1"]), float(o["x2"])
            y1, y2 = float(o["y1"]), float(o["y2"])
            xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
            ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)

            pts = np.array([
                [xa, ya, 0.0],
                [xb, ya, 0.0],
                [xb, yb, 0.0],
                [xa, yb, 0.0],
            ], dtype=float)

            faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
            rect = pv.PolyData(pts, faces)
            rect["id"] = np.full(rect.n_points, i, dtype=np.int32)
            rect["material"] = np.array([mat] * rect.n_points, dtype=object)
            blocks.append(rect)

    blocks.save(str(out_dir / "objects.vtm"))


def draw_png(out_path: Path, U: np.ndarray, scene: dict, title: str):
    src = scene["scene"]["source"]
    x0 = float(src.get("x0", 0.8))
    y0 = float(src.get("y0", 1.0))

    plt.figure(figsize=(7, 5))
    plt.imshow(U, origin="lower", extent=[0, 3, 0, 2], aspect="auto")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    # objects overlay
    for i, o in enumerate(scene["scene"].get("objects", []), start=1):
        t = str(o.get("type", "rect")).lower()
        mat = str(o.get("material", "air"))
        if t == "circle":
            cx = float(o["cx"]); cy = float(o["cy"]); r = float(o["r"])
            th = np.linspace(0, 2*np.pi, 200)
            plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), color="black", linewidth=1.2)
            plt.text(cx + r + 0.02, cy + r + 0.02, f"{i}:{mat}", color="black", fontsize=9)
        else:
            x1 = float(o["x1"]); x2 = float(o["x2"])
            y1 = float(o["y1"]); y2 = float(o["y2"])
            xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
            ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
            plt.plot([xa, xb, xb, xa, xa], [ya, ya, yb, yb, ya], color="black", linewidth=1.2)
            plt.text(xa + 0.02, yb + 0.02, f"{i}:{mat}", color="black", fontsize=9)

    plt.scatter([x0], [y0], s=120, c="red", edgecolors="black", linewidths=0.5, zorder=5)
    plt.text(x0 + 0.05, y0, "Source", color="black", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(scene_path: str, epochs: int):
    scene_path = Path(scene_path).resolve()
    scene = load_scene(scene_path)

    objs = scene["scene"].get("objects", [])
    print(f"[INFO] device = {choose_device()}")
    print(f"[INFO] scene = {scene_path}")
    print(f"[INFO] objects = {len(objs)}")

    device = choose_device()

    out_root = scene_path.parent / "simple_results"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, X, Y = make_grid(nx=360, ny=240, xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)
    U_true = field_true_numpy(scene, X, Y)

    # training points
    rng = np.random.default_rng(0)
    n_train = 18000
    ix = rng.integers(0, X.shape[1], size=n_train)
    iy = rng.integers(0, X.shape[0], size=n_train)
    x_train = np.stack([X[iy, ix], Y[iy, ix]], axis=1).astype(np.float32)
    y_train = U_true[iy, ix].reshape(-1, 1).astype(np.float32)

    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)

    model = MLP(in_dim=2, width=128, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

        if ep == 1 or ep % 100 == 0 or ep == epochs:
            dt = time.time() - t0
            print(f"Epoch {ep:4d}/{epochs}  MSE={loss.item():.6e}  time={dt:.1f}s")

    # full prediction
    xy_full = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
    with torch.no_grad():
        U_pred = model(torch.from_numpy(xy_full).to(device)).float().cpu().numpy().reshape(Y.shape).astype(np.float32)

    U_err = np.abs(U_pred - U_true).astype(np.float32)

    mse = float(np.mean((U_pred - U_true) ** 2))
    mae = float(np.mean(np.abs(U_pred - U_true)))
    maxe = float(np.max(np.abs(U_pred - U_true)))

    (out_dir / "metrics.txt").write_text(
        f"device={device} | objects={len(objs)} | epochs={epochs} | MSE={mse:.6e} | MAE={mae:.6e} | MAX_ERR={maxe:.6e}\n",
        encoding="utf-8"
    )

    # VTI export (if pyvista)
    export_vti(out_dir / "U_true.vti", U_true, xs, ys, name="U_true")
    export_vti(out_dir / "U_pred.vti", U_pred, xs, ys, name="U_pred")
    export_vti(out_dir / "U_err.vti",  U_err,  xs, ys, name="U_err")

    # PNG export
    draw_png(out_dir / "field_true.png", U_true, scene, "Field true")
    draw_png(out_dir / "field_pred.png", U_pred, scene, "Field pred")
    draw_png(out_dir / "field_err.png",  U_err,  scene, "Field err")

    # Objects export (ParaView)
    save_objects_vtm(out_dir, scene)

    print(f"[OK] Saved results: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="path to scene.yaml")
    ap.add_argument("--epochs", type=int, default=2000)
    args = ap.parse_args()
    epochs = int(max(2000, min(200000, args.epochs)))
    main(args.scene, epochs=epochs)
