import argparse, subprocess, sys, re
from pathlib import Path
import yaml
import numpy as np

WORLD = dict(xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)

DEFAULT_MATERIALS = {
    "air":      {"absorption": 0.03, "c": 3.0e8},
    "metal":    {"absorption": 5.0,  "c": 2.5e8},
    "plastic":  {"absorption": 0.35, "c": 2.4e8},
    "glass":    {"absorption": 0.6,  "c": 2.2e8},
    "concrete": {"absorption": 2.5,  "c": 2.0e8},
    "water":    {"absorption": 0.5,  "c": 1.5e8},
    "wood":     {"absorption": 1.0,  "c": 2.0e8},
    "rubber":   {"absorption": 3.0,  "c": 1.6e8},
}

DEFAULT_SOURCE = dict(x0=0.8, y0=1.0, amplitude=1.0, frequency_hz=1e9)

def run_solver(solver: Path, scene: Path, extra: list[str], cwd: Path) -> Path:
    cmd = [sys.executable, str(solver), str(scene)] + extra
    print("\nRUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    print(out)
    if proc.returncode != 0:
        raise RuntimeError(f"Solver failed: {proc.returncode}")
    m = re.search(r"Saved results:\s*(.+)", out)
    if not m:
        raise RuntimeError("No 'Saved results:' line found")
    out_dir = Path(m.group(1).strip())
    if not out_dir.is_absolute():
        out_dir = (cwd / out_dir).resolve()
    return out_dir

def save_scene(path: Path, objects: list[dict], materials=None, source=None):
    data = {
        "materials": materials if materials is not None else DEFAULT_MATERIALS,
        "scene": {"source": source if source is not None else DEFAULT_SOURCE,
                  "objects": objects}
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def load_npy(out_dir: Path, name: str) -> np.ndarray:
    p = out_dir / name
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Add np.save(...) in solver.")
    return np.load(p)

def radial_residual(U: np.ndarray, x0: float, y0: float) -> float:
    # measure how well U depends only on radius from (x0,y0)
    ny, nx = U.shape
    gx = np.linspace(WORLD["xmin"], WORLD["xmax"], nx, dtype=np.float32)
    gy = np.linspace(WORLD["ymin"], WORLD["ymax"], ny, dtype=np.float32)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)

    rmax = float(R.max())
    bins = 200
    rb = np.clip((R / (rmax + 1e-12) * (bins - 1)).astype(np.int32), 0, bins - 1)

    prof = np.zeros((bins,), dtype=np.float64)
    cnt = np.zeros((bins,), dtype=np.float64)
    flatU = U.astype(np.float64).ravel()
    flatB = rb.ravel()
    np.add.at(prof, flatB, flatU)
    np.add.at(cnt, flatB, 1.0)
    prof /= (cnt + 1e-12)

    U_hat = prof[rb]
    resid = np.sqrt(np.mean((U - U_hat.astype(np.float32))**2))
    return float(resid)

def rms(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="solver_simple_torch.py")
    ap.add_argument("--workdir", default=".")
    args = ap.parse_args()

    solver = Path(args.solver).resolve()
    cwd = Path(args.workdir).resolve()
    scene = cwd / "scene_test.yaml"

    ok_all = True

    # 1) no objects -> radial symmetry around source
    save_scene(scene, [])
    out = run_solver(solver, scene, ["--epochs","2000"], cwd)
    U = load_npy(out, "U_pred.npy")
    res = radial_residual(U, DEFAULT_SOURCE["x0"], DEFAULT_SOURCE["y0"])
    ok = res < 0.15  # heuristic; depends on your smoothing
    print("[PASS]" if ok else "[FAIL]", "No objects radial residual", res)
    ok_all &= ok

    # 2) absorption effect
    objs = [{"x":1.6,"y":1.0,"r":0.10,"material":"wood"}]
    mats_low = dict(DEFAULT_MATERIALS);  mats_low["wood"] = dict(mats_low["wood"]);  mats_low["wood"]["absorption"]=0.05
    mats_high= dict(DEFAULT_MATERIALS);  mats_high["wood"]= dict(mats_high["wood"]); mats_high["wood"]["absorption"]=5.0

    save_scene(scene, objs, materials=mats_low)
    outL = run_solver(solver, scene, ["--epochs","2000"], cwd)
    UL = load_npy(outL, "U_pred.npy")

    save_scene(scene, objs, materials=mats_high)
    outH = run_solver(solver, scene, ["--epochs","2000"], cwd)
    UH = load_npy(outH, "U_pred.npy")

    ok = float(np.std(UH)) < float(np.std(UL)) * 0.98
    print("[PASS]" if ok else "[FAIL]", "Absorption higher -> lower std",
          "std_low", float(np.std(UL)), "std_high", float(np.std(UH)))
    ok_all &= ok

    # 4) transmission effect (use args)
    save_scene(scene, [{"x":1.6,"y":1.0,"r":0.12,"material":"glass"}])
    out_hi = run_solver(solver, scene, ["--epochs","2000","--trans_glass","0.95"], cwd)
    out_lo = run_solver(solver, scene, ["--epochs","2000","--trans_glass","0.20"], cwd)
    Uhi = load_npy(out_hi, "U_pred.npy")
    Ulo = load_npy(out_lo, "U_pred.npy")
    ok = float(np.std(Ulo)) < float(np.std(Uhi)) * 0.98
    print("[PASS]" if ok else "[FAIL]", "Transmission lower -> lower std",
          "std_hi", float(np.std(Uhi)), "std_lo", float(np.std(Ulo)))
    ok_all &= ok

    # 6) diffraction toggle: disable spectral smoothing to see effect clearly
    save_scene(scene, [{"x":1.6,"y":1.0,"r":0.18,"material":"concrete"}])
    out_on = run_solver(solver, scene, ["--epochs","2000","--edge_amp","0.50","--spectral_mix","0.0"], cwd)
    out_off= run_solver(solver, scene, ["--epochs","2000","--edge_amp","0.0","--spectral_mix","0.0"], cwd)
    Uon = load_npy(out_on, "U_pred.npy")
    Uoff= load_npy(out_off,"U_pred.npy")
    d = rms(Uon,Uoff)
    ok = d > 0.01
    print("[PASS]" if ok else "[FAIL]", "Diffraction edge_amp effect rms", d)
    ok_all &= ok

    print("\nFINAL:", "ALL PASS ✅" if ok_all else "SOME FAIL ❌")

if __name__ == "__main__":
    main()
