import yaml
from pathlib import Path

def main():
    p = Path("scene.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    f = data["scene"]["source"]["frequency_hz"]
    n = len(data["scene"]["objects"])
    mats = list(data["materials"].keys())
    print(f"[OK] scene.yaml loaded: f={f} Hz, objects={n}, materials={mats}")
    Path("quick_test_ok.txt").write_text("OK\n", encoding="utf-8")
    print("[OK] quick_test_ok.txt written")

if __name__ == "__main__":
    main()
