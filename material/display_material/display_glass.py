import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

filename = "/workspace/my_examples/example_antenna/outputs/glass/constraints/wave_constraint.vtp"
mesh = pv.read(filename)

points = mesh.points
x, y = points[:, 0], points[:, 1]

true_u = mesh["true_u"]
pred_u = mesh["pred_u"]

nx = ny = 500
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
X, Y = np.meshgrid(xi, yi, indexing="ij")

U_true = griddata((x, y), true_u, (X, Y), method="cubic")
U_pred = griddata((x, y), pred_u, (X, Y), method="cubic")

grid = pv.ImageData()
grid.origin = (x.min(), y.min(), 0)
grid.spacing = ((x.max()-x.min())/(nx-1), (y.max()-y.min())/(ny-1), 1)
grid.dimensions = (nx, ny, 1)

grid["true_u"] = U_true.ravel(order="F")
grid["pred_u"] = U_pred.ravel(order="F")

pv.MultiBlock([grid]).save(
    "/workspace/my_examples/example_antenna/outputs/glass/constraints/wave_field_structured.vtm"
)
