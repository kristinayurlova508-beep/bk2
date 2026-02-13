#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import yaml

WORLD = dict(xmin=0.0, xmax=3.0, ymin=0.0, ymax=2.0)

DEFAULT_MATERIALS = {
    "air":      {"absorption": 0.03, "c": 3.0e8},
    "asphalt":  {"absorption": 0.06, "c": 2.2e8},
    "brick":    {"absorption": 0.07, "c": 2.0e8},
    "concrete": {"absorption": 0.08, "c": 2.0e8},
    "foam":     {"absorption": 0.05, "c": 1.3e8},
    "glass":    {"absorption": 0.05, "c": 2.2e8},
    "ice":      {"absorption": 0.02, "c": 3.1e8},
    "metal":    {"absorption": 0.10, "c": 2.5e8},
    "plastic":  {"absorption": 0.05, "c": 2.4e8},
    "rubber":   {"absorption": 0.09, "c": 1.6e8},
    "sand":     {"absorption": 0.08, "c": 1.7e8},
    "water":    {"absorption": 0.06, "c": 1.5e8},
    "wood":     {"absorption": 0.06, "c": 2.0e8},
}

def die(msg: str):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)

def fnum(x: str) -> float:
    # strict float parse
    try:
        return float(x)
    except Exception:
        die(f"not a number: {x}")

def in_range(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def load_scene(path: Path) -> dict:
    if not path.exists():
        # init default scene
        return {
            "materials": DEFAULT_MATERIALS,
            "scene": {
                "source": {"x0": 0.8, "y0": 1.0, "frequency_hz": 1e9, "amplitude": 1.0},
                "objects": []
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "scene" not in data:
        die("scene.yaml missing top-level key: scene")
    data.setdefault("materials", DEFAULT_MATERIALS)
    data["scene"].setdefault("source", {"x0": 0.8, "y0": 1.0, "frequency_hz": 1e9, "amplitude": 1.0})
    data["scene"].setdefault("objects", [])
    if not isinstance(data["scene"]["objects"], list):
        data["scene"]["objects"] = []
    return data

def save_scene(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def list_objects(data: dict):
    objs = data["scene"]["objects"]
    if not objs:
        print("(no objects)")
        return
    for i, o in enumerate(objs):
        x = o.get("x", 0.0)
        y = o.get("y", 0.0)
        r = o.get("r", 0.08)
        m = o.get("material", "air")
        print(f"{i:02d}: x={x:.4f} y={y:.4f} r={r:.4f} material={m}")

def add_object(data: dict, x: float, y: float, material: str, r: float):
    material = material.strip().lower()
    if material not in DEFAULT_MATERIALS:
        die(f"unknown material: {material}. Allowed: {', '.join(sorted(DEFAULT_MATERIALS.keys()))}")
    x = in_range(x, WORLD["xmin"], WORLD["xmax"])
    y = in_range(y, WORLD["ymin"], WORLD["ymax"])
    r = abs(r)
    data["scene"]["objects"].append({"x": x, "y": y, "r": r, "material": material})

def delete_object(data: dict, idx: int):
    objs = data["scene"]["objects"]
    if idx < 0 or idx >= len(objs):
        die(f"bad index {idx}. Use: list")
    del objs[idx]

def update_object(data: dict, idx: int, x, y, material, r):
    objs = data["scene"]["objects"]
    if idx < 0 or idx >= len(objs):
        die(f"bad index {idx}. Use: list")
    o = objs[idx]
    if x is not None: o["x"] = in_range(x, WORLD["xmin"], WORLD["xmax"])
    if y is not None: o["y"] = in_range(y, WORLD["ymin"], WORLD["ymax"])
    if r is not None: o["r"] = abs(r)
    if material is not None:
        material = material.strip().lower()
        if material not in DEFAULT_MATERIALS:
            die(f"unknown material: {material}")
        o["material"] = material

def set_source(data: dict, x0, y0, freq, amp):
    s = data["scene"]["source"]
    if x0 is not None: s["x0"] = in_range(x0, WORLD["xmin"], WORLD["xmax"])
    if y0 is not None: s["y0"] = in_range(y0, WORLD["ymin"], WORLD["ymax"])
    if freq is not None: s["frequency_hz"] = float(freq)
    if amp is not None: s["amplitude"] = float(amp)

def main():
    ap = argparse.ArgumentParser(description="No-GUI scene editor for scene.yaml")
    ap.add_argument("--scene", default="scene.yaml", help="path to scene.yaml")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list objects")

    a = sub.add_parser("add", help="add object")
    a.add_argument("--x", required=True)
    a.add_argument("--y", required=True)
    a.add_argument("--material", required=True)
    a.add_argument("--r", default="0.08")

    d = sub.add_parser("del", help="delete object by index")
    d.add_argument("index", type=int)

    u = sub.add_parser("update", help="update object by index")
    u.add_argument("index", type=int)
    u.add_argument("--x")
    u.add_argument("--y")
    u.add_argument("--material")
    u.add_argument("--r")

    s = sub.add_parser("source", help="set source parameters")
    s.add_argument("--x0")
    s.add_argument("--y0")
    s.add_argument("--freq")
    s.add_argument("--amp")

    args = ap.parse_args()
    scene_path = Path(args.scene)

    data = load_scene(scene_path)

    if args.cmd == "list":
        list_objects(data)
        return

    if args.cmd == "add":
        add_object(
            data,
            x=fnum(args.x),
            y=fnum(args.y),
            material=args.material,
            r=fnum(args.r),
        )
        save_scene(scene_path, data)
        print("OK added. Now:")
        list_objects(data)
        return

    if args.cmd == "del":
        delete_object(data, args.index)
        save_scene(scene_path, data)
        print("OK deleted. Now:")
        list_objects(data)
        return

    if args.cmd == "update":
        update_object(
            data, args.index,
            x=fnum(args.x) if args.x is not None else None,
            y=fnum(args.y) if args.y is not None else None,
            material=args.material,
            r=fnum(args.r) if args.r is not None else None
        )
        save_scene(scene_path, data)
        print("OK updated. Now:")
        list_objects(data)
        return

    if args.cmd == "source":
        set_source(
            data,
            x0=fnum(args.x0) if args.x0 is not None else None,
            y0=fnum(args.y0) if args.y0 is not None else None,
            freq=fnum(args.freq) if args.freq is not None else None,
            amp=fnum(args.amp) if args.amp is not None else None,
        )
        save_scene(scene_path, data)
        print("OK source updated:", data["scene"]["source"])
        return

if __name__ == "__main__":
    main()
