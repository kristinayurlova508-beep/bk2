import numpy as np, glob, os

def test_radial_symmetry_on_empty_scene():
    out = sorted(glob.glob("simple_results/*"))[-1]
    U = np.load(os.path.join(out, "U.npy"))
    ny, nx = U.shape

    # берём несколько углов на одном радиусе
    cy, cx = ny//2, nx//2
    r = min(nx, ny)//6
    samples = []
    for ang in np.linspace(0, 2*np.pi, 16, endpoint=False):
        x = int(cx + r*np.cos(ang))
        y = int(cy + r*np.sin(ang))
        samples.append(U[y, x])
    samples = np.array(samples)
    assert samples.std() < 0.1 * (np.abs(samples).mean() + 1e-6)
