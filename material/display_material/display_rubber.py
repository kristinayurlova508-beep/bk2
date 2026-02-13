import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

filename = "/workspace/my_examples/example_antenna/outputs/rubber/constraints/wave_constraint.vtp"
mesh = pv.read(filename)

pts = mesh.points
x, y = pts[:,0], pts[:,1]

nx = ny = 500
X, Y = np.meshgrid(
    np.linspace(x.min(), x.max(), nx),
    np.linspace(y.min(), y.max(), ny),
    indexing="ij"
)

U = griddata((x, y), mesh["pred_u"], (X, Y), method="cubic")

grid = pv.ImageData()
grid.origin = (x.min(), y.min(), 0)
grid.spacing = ((x.max()-x.min())/(nx-1), (y.max()-y.min())/(ny-1), 1)
grid.dimensions = (nx, ny, 1)

grid["u"] = U.ravel(order="F")

pv.MultiBlock([grid]).save(
    "/workspace/my_examples/example_antenna/outputs/rubber/constraints/wave_field_structured.vtm"
)
