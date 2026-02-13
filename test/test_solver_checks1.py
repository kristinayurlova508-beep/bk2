import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import re
import math

import yaml
import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Need pillow: pip install pillow") from e


WORLD = dict(xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)


DEFAULT_MATERIALS = {
    # You can keep your GUI-like defaults; solver reads "absorption" and optional "c"
    "air":      {"absorption": 0.03, "c": 3.0e8},
    "metal":    {"absorption": 5.0,  "c": 2.5e8},
    "plastic":  {"absorption": 0.35, "c": 2.4e8},
    "glass":    {"absorption": 0.6,  "c": 2.2e8},
    "concrete": {"absorption": 2.5,  "c": 2.0e8},
    "water":    {"absorption": 0.5,  "c": 1.5e8},
    "wood":     {"absorption": 1.0,  "c": 2.0e8},
    "rubber":   {"absorption": 3.0,  "c": 1.6e8},
    "sand":     {"absorption": 1.8,  "c": 1.7e8},
    "asphalt":  {"absorption": 3.2,  "c": 2.2e8},
    "foam":     {"absorption": 4.5,  "c": 1.3e8},
    "ice":      {"absorption": 0.2,  "c": 3.1e8},
    "brick":    {"absorption": 2.2,  "c": 2.0e8},
}

DEFAULT_SOURCE = dict(x0=0.8, y0=1.0, amplitude=1.0, frequency_hz=1e9)


# -----------------------------
# helpers
# -----------------------------
def run_solver(solver: Path, scene_path: Path, extra_args: list[str], cwd: Path) -> Path:
    cmd = [sys.executable, str(solver), str(scene_path)] + extra_args
    print("\nRUN:", " ".join(cmd))

    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    print(out)

    if proc.returncode != 0:
        raise RuntimeError(f"Solver failed with code {proc.returncode}")

    m = re.search(r"Saved results:\s*(.+)", out)
    if not m:
        raise RuntimeError("Cannot find 'Saved results: ...' in solver output")
    out_dir = Path(m.group(1).strip())
    if not out_dir.exists():
        # if solver prints relative path, resolve from cwd
        out_dir = (cwd / out_dir).resolve()
    if not out_dir.exists():
        raise RuntimeError(f"Results folder not found: {out_dir}")
    return out_dir


def save_scene(path: Path, objects: list[dict], materials: dict | None = None, source: dict | None = None):
    data = {
        "materials": materials if materials is not None else DEFAULT_MATERIALS,
        "scene": {
            "source": source if source is not None else DEFAULT_SOURCE,
            "objects": objects,
        }
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_field_png(png_path: Path) -> np.ndarray:
    im = Image.open(png_path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    # convert to grayscale intensity (good enough for comparisons)
    g = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return g


def radial_symmetry_score(img: np.ndarray) -> float:
    """
    Measures how similar image is to its left-right and up-down flips.
    1.0 = perfectly symmetric, lower = less symmetric
    """
    a = img
    lr = np.flip(a, axis=1)
    ud = np.flip(a, axis=0)

    def corr(x, y):
        x = x.flatten()
        y = y.flatten()
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
        return float(np.dot(x, y) / denom)

    return 0.5 * (corr(a, lr) + corr(a, ud))


def diff_norm(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float32)
    return float(np.sqrt(np.mean(d * d)))


def mean_intensity(img: np.ndarray) -> float:
    return float(img.mean())


def print_check(name: str, ok: bool, info: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {('- ' + info) if info else ''}")


# -----------------------------
# tests
# -----------------------------
def test_no_objects_symmetry(solver: Path, workdir: Path) -> bool:
    scene = workdir / "scene_test.yaml"
    save_scene(scene, objects=[])

    out_dir = run_solver(solver, scene, extra_args=["--epochs", "2000"], cwd=workdir)
    img = load_field_png(out_dir / "field_pred.png")

    score = radial_symmetry_score(img)
    # expect high symmetry when no objects
    ok = score > 0.90
    print_check("No objects -> symmetry", ok, f"score={score:.3f} (want > 0.90)")
    return ok


def test_absorption_effect(solver: Path, workdir: Path) -> bool:
    # same geometry, only absorption differs
    scene = workdir / "scene_test.yaml"
    objs = [{"x": 1.6, "y": 1.0, "r": 0.10, "material": "wood"}]
    mats_low = dict(DEFAULT_MATERIALS)
    mats_high = dict(DEFAULT_MATERIALS)
    mats_low["wood"] = dict(mats_low["wood"]);  mats_low["wood"]["absorption"] = 0.05
    mats_high["wood"] = dict(mats_high["wood"]); mats_high["wood"]["absorption"] = 5.0

    save_scene(scene, objects=objs, materials=mats_low)
    out_low = run_solver(solver, scene, extra_args=["--epochs", "2000"], cwd=workdir)
    img_low = load_field_png(out_low / "field_pred.png")

    save_scene(scene, objects=objs, materials=mats_high)
    out_high = run_solver(solver, scene, extra_args=["--epochs", "2000"], cwd=workdir)
    img_high = load_field_png(out_high / "field_pred.png")

    # more absorption -> typically darker / lower intensity patterns
    mlow = mean_intensity(img_low)
    mhigh = mean_intensity(img_high)
    ok = mhigh < mlow * 0.98  # allow small tolerance
    print_check("Absorption effect (higher -> darker)", ok, f"mean_low={mlow:.3f}, mean_high={mhigh:.3f}")
    return ok


def test_speed_c_phase_effect(solver: Path, workdir: Path) -> bool:
    scene = workdir / "scene_test.yaml"
    objs = [{"x": 1.6, "y": 1.0, "r": 0.10, "material": "glass"}]

    mats_fast = dict(DEFAULT_MATERIALS)
    mats_slow = dict(DEFAULT_MATERIALS)
    mats_fast["glass"] = dict(mats_fast["glass"]); mats_fast["glass"]["c"] = 3.0e8
    mats_slow["glass"] = dict(mats_slow["glass"]); mats_slow["glass"]["c"] = 1.2e8

    save_scene(scene, objects=objs, materials=mats_fast)
    out_fast = run_solver(solver, scene, extra_args=["--epochs", "2000"], cwd=workdir)
    img_fast = load_field_png(out_fast / "field_pred.png")

    save_scene(scene, objects=objs, materials=mats_slow)
    out_slow = run_solver(solver, scene, extra_args=["--epochs", "2000"], cwd=workdir)
    img_slow = load_field_png(out_slow / "field_pred.png")

    d = diff_norm(img_fast, img_slow)
    ok = d > 0.01  # should noticeably change phase/pattern
    print_check("Speed c changes phase/pattern", ok, f"rms_diff={d:.4f} (want > 0.01)")
    return ok


def test_transmission_effect(solver: Path, workdir: Path) -> bool:
    # same, but use trans_* args (affects rays through material)
    scene = workdir / "scene_test.yaml"
    objs = [{"x": 1.6, "y": 1.0, "r": 0.12, "material": "glass"}]
    save_scene(scene, objects=objs)

    out_hi = run_solver(
        solver, scene,
        extra_args=["--epochs", "2000", "--trans_glass", "0.95"],
        cwd=workdir
    )
    img_hi = load_field_png(out_hi / "field_pred.png")

    out_lo = run_solver(
        solver, scene,
        extra_args=["--epochs", "2000", "--trans_glass", "0.20"],
        cwd=workdir
    )
    img_lo = load_field_png(out_lo / "field_pred.png")

    mhi = mean_intensity(img_hi)
    mlo = mean_intensity(img_lo)
    ok = mlo < mhi * 0.98
    print_check("Transmission lower -> darker", ok, f"mean_hi={mhi:.3f}, mean_lo={mlo:.3f}")
    return ok


def test_shadow_barrier_effect(solver: Path, workdir: Path) -> bool:
    scene = workdir / "scene_test.yaml"
    objs = [{"x": 1.6, "y": 1.0, "r": 0.16, "material": "glass"}]
    save_scene(scene, objects=objs)

    # make glass a barrier by adding it to barrier_materials
    out_bar = run_solver(
        solver, scene,
        extra_args=["--epochs", "2000", "--barrier_materials", "metal", "concrete", "glass"],
        cwd=workdir
    )
    img_bar = load_field_png(out_bar / "field_pred.png")

    # not a barrier (default)
    out_nobar = run_solver(
        solver, scene,
        extra_args=["--epochs", "2000"],
        cwd=workdir
    )
    img_nobar = load_field_png(out_nobar / "field_pred.png")

    d = diff_norm(img_bar, img_nobar)
    ok = d > 0.01
    print_check("Barrier list affects shadow", ok, f"rms_diff={d:.4f} (want > 0.01)")
    return ok


def test_diffraction_toggle(solver: Path, workdir: Path) -> bool:
    scene = workdir / "scene_test.yaml"
    objs = [{"x": 1.6, "y": 1.0, "r": 0.18, "material": "concrete"}]
    save_scene(scene, objects=objs)

    out_on = run_solver(solver, scene, extra_args=["--epochs", "2000", "--edge_amp", "0.12"], cwd=workdir)
    img_on = load_field_png(out_on / "field_pred.png")

    out_off = run_solver(solver, scene, extra_args=["--epochs", "2000", "--edge_amp", "0.0"], cwd=workdir)
    img_off = load_field_png(out_off / "field_pred.png")

    d = diff_norm(img_on, img_off)
    ok = d > 0.01
    print_check("Diffraction edge_amp changes field", ok, f"rms_diff={d:.4f} (want > 0.01)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", type=str, default="solver_simple_torch.py",
                    help="path to your solver .py")
    ap.add_argument("--workdir", type=str, default=".",
                    help="project dir where solver writes simple_results/")
    ap.add_argument("--keep", action="store_true",
                    help="keep test_runs folder")
    args = ap.parse_args()

    solver = Path(args.solver).resolve()
    workdir = Path(args.workdir).resolve()

    if not solver.exists():
        raise SystemExit(f"Solver not found: {solver}")

    test_root = workdir / "test_runs"
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    # Force solver output folders to stay; we just run and then copy results under test_runs
    results_keep = []

    all_ok = True

    def run_and_store(test_fn, name):
        nonlocal all_ok
        print("\n" + "=" * 70)
        print("TEST:", name)
        ok = test_fn(solver, workdir)
        all_ok = all_ok and ok

        # copy latest simple_results folder mentioned in console is hard;
        # but solver already stores time-based folder. We don't need to copy.
        return ok

    run_and_store(test_no_objects_symmetry, "1) No objects -> symmetry")
    run_and_store(test_absorption_effect, "2) Absorption changes amplitude")
    run_and_store(test_speed_c_phase_effect, "3) Speed c changes phase")
    run_and_store(test_transmission_effect, "4) Transmission changes amplitude")
    run_and_store(test_shadow_barrier_effect, "5) Barrier list changes shadow")
    run_and_store(test_diffraction_toggle, "6) edge_amp toggles diffraction")

    print("\n" + "=" * 70)
    print("FINAL:", "ALL PASS ✅" if all_ok else "SOME FAIL ❌")
    print("Check images in your solver output folder simple_results/ (each test run makes a new timestamp folder).")


if __name__ == "__main__":
    main()
