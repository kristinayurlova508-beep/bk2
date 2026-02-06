import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
import pandas as pd

filename = "/workspace/my_examples/example_antenna/outputs/plastic/constraints/wave_constraint.vtp"
mesh = pv.read(filename)

points = mesh.points
x_data = points[:, 0]
y_data = points[:, 1]

true_u_data = mesh["true_u"]
pred_u_data = mesh["pred_u"]

x_min, x_max = x_data.min(), x_data.max()
y_min, y_max = y_data.min(), y_data.max()

nx, ny = 500, 500
grid_x = np.linspace(x_min, x_max, nx)
grid_y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")

true_u_grid = griddata((x_data, y_data), true_u_data, (X, Y), method="cubic")
pred_u_grid = griddata((x_data, y_data), pred_u_data, (X, Y), method="cubic")

diff_grid = true_u_grid - pred_u_grid

grid = pv.ImageData()
grid.origin = (x_min, y_min, 0.0)
grid.spacing = ((x_max - x_min)/(nx-1), (y_max - y_min)/(ny-1), 1.0)
grid.dimensions = (nx, ny, 1)

grid["true_u"] = np.ravel(true_u_grid, order="F")
grid["pred_u"] = np.ravel(pred_u_grid, order="F")
grid["difference"] = np.ravel(diff_grid, order="F")

antenna = pv.Sphere(radius=0.04, center=(1, 1, 0))
pv.MultiBlock([grid, antenna]).save(
    "/workspace/my_examples/example_antenna/outputs/plastic/constraints/wave_field_structured_antenna.vtm"
)
