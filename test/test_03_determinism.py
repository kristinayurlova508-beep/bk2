# tests/test_03_determinism.py
import subprocess, sys, glob, os, hashlib, time

def sha256(p):
    h = hashlib.sha256()
    h.update(open(p, "rb").read())
    return h.hexdigest()

def test_deterministic_outputs(tmp_path):
    scene = tmp_path/"scene.yaml"
    scene.write_text("""
scene:
  source: {x0: 0.8, y0: 1.0, frequency_hz: 6e9, amplitude: 1.0}
  objects: []
""", encoding="utf-8")

    subprocess.check_call([sys.executable, "solver_simple_torch.py", str(scene), "--nx", "120", "--ny", "80"])
    out1 = sorted(glob.glob("simple_results/*"))[-1]
    h1 = sha256(os.path.join(out1, "field_pred.png"))

    time.sleep(0.1)  # чтобы новая папка отличалась по времени
    subprocess.check_call([sys.executable, "solver_simple_torch.py", str(scene), "--nx", "120", "--ny", "80"])
    out2 = sorted(glob.glob("simple_results/*"))[-1]
    h2 = sha256(os.path.join(out2, "field_pred.png"))

    assert h1 == h2
