# tests/test_02_metrics_sane.py
import glob, os, re

def test_metrics_vmax_reasonable():
    out_dir = sorted(glob.glob("simple_results/*"))[-1]
    txt = open(os.path.join(out_dir, "metrics.txt"), encoding="utf-8").read()
    m = re.search(r"vmax=(.+)", txt)
    assert m, "vmax not found"
    vmax = float(m.group(1))
    assert vmax == vmax  # not NaN
    assert vmax < 1e6    # очень грубый, но полезный предел
