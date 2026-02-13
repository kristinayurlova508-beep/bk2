import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
import pandas as pd


# --------------------------------------------------------
# 1) Čítanie existujúceho VTP súboru
# --------------------------------------------------------
filename = "/workspace/my_examples/example_antenna/outputs/water/constraints/wave_constraint.vtp"  # uprav cestu, ak je potrebné
mesh = pv.read(filename)

# Extrahovanie bodových súradníc (x, y, z)
points = mesh.points  # tvar: (N, 3)
x_data = points[:, 0]
y_data = points[:, 1]

# Zoznam dostupných polí
print("Available arrays:", mesh.array_names)
true_u_data = mesh["true_u"]
pred_u_data = mesh["pred_u"]

# --------------------------------------------------------
# 2) Interpolácia do pravidelnej 2D mriežky (x vodorovne, y zvislo)
# --------------------------------------------------------
x_min, x_max = x_data.min(), x_data.max()
y_min, y_max = y_data.min(), y_data.max()

# Zvýšenie rozlíšenia pre ostrejší výstup
nx, ny = 500, 500  # vyššie rozlíšenie
grid_x = np.linspace(x_min, x_max, nx)  # os x
grid_y = np.linspace(y_min, y_max, ny)  # os y

# Použitie 'ij' indexovania, aby prvá dimenzia bola x a druhá y
X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")

true_u_grid = griddata((x_data, y_data), true_u_data, (X, Y), method="cubic")
pred_u_grid = griddata((x_data, y_data), pred_u_data, (X, Y), method="cubic")

# Vypočítanie rozdielového poľa
diff_grid = true_u_grid - pred_u_grid
# Alebo ak chcete absolútny rozdiel:
# diff_grid = np.abs(true_u_grid - pred_u_grid)

# --------------------------------------------------------
# 3) Vytvorenie štruktúrovanej mriežky pomocou ImageData a pripojenie dát
# --------------------------------------------------------
grid = pv.ImageData()
grid.origin = (x_min, y_min, 0.0)  # ľavý dolný roh
grid.spacing = ((x_max - x_min) / (nx - 1),
                (y_max - y_min) / (ny - 1),
                1.0)
grid.dimensions = (nx, ny, 1)

# "Rozbalenie" 2D polí v stĺpcovom poradí ("F") tak, aby zodpovedali pamäťovému usporiadaniu vo VTK
grid["true_u"] = np.ravel(true_u_grid, order="F")
grid["pred_u"] = np.ravel(pred_u_grid, order="F")
grid["difference"] = np.ravel(diff_grid, order="F")

mask = ~np.isnan(true_u_grid) & ~np.isnan(pred_u_grid)
error = true_u_grid[mask] - pred_u_grid[mask]
abs_error = np.abs(error)

mae = np.mean(abs_error)
rmse = np.sqrt(np.mean(error ** 2))
max_error = np.max(abs_error)

# Save to CSV
metrics = {
    "MAE": [mae],
    "RMSE": [rmse],
    "MaxError": [max_error]
}

# Optionally, add metadata like iteration number
# Example: metrics["Iteration"] = [5000]

df = pd.DataFrame(metrics)
print(df)

# --------------------------------------------------------
# 4) Pridanie bodu (guľa) reprezentujúcej polohu antény
# --------------------------------------------------------
# Nastavte polohu antény – napr. (1, 1, 0)
antenna_pos = (1, 1, 0)
antenna = pv.Sphere(radius=0.04, center=antenna_pos)

# --------------------------------------------------------
# 5) Kombinácia gridu a antény do MultiBlock datasetu a uloženie do súboru
# --------------------------------------------------------
multiblock_antenna = pv.MultiBlock([grid, antenna])
output_filename1 = "/workspace/my_examples/example_antenna/outputs/water/constraints/wave_field_structured_antenna.vtm"
multiblock_antenna.save(output_filename1)

multiblock = pv.MultiBlock([grid])
output_filename2 = "/workspace/my_examples/example_antenna/outputs/water/constraints/wave_field_structured.vtm"
multiblock.save(output_filename2)
print(f"Structured grid with difference and antenna marker saved to: {output_filename2}")
