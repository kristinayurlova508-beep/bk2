import os
import numpy as np
import torch
import meshio
from modulus.sym.hydra import to_absolute_path

def run_inference(
    wave_network_node,
    device,
    x_min=0.0, x_max=2.0,
    y_min=0.0, y_max=1.0,
    nx=100,
    ny=100,
    t_min=0.0,
    t_max=2.0,
    num_time_steps=50,
    output_prefix="wave_solution"
):
    x_vals = np.linspace(x_min, x_max, nx)
    y_vals = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    t_vals = np.linspace(t_min, t_max, num_time_steps)
    
    for i, t_val in enumerate(t_vals):
        invar_numpy = {
            "x": X.flatten(),
            "y": Y.flatten(),
            "t": np.full(X.size, t_val, dtype=np.float32),
        }
        invar_torch = {
            k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(1)
            for k, v in invar_numpy.items()
        }
        outvar = wave_network_node.evaluate(invar_torch)
        U = outvar["u"].detach().cpu().numpy().reshape((ny, nx))
        
        points = np.column_stack([X.flatten(), Y.flatten(), np.zeros(X.flatten().shape)])
        quads = []
        for row in range(ny - 1):
            for col in range(nx - 1):
                idx = row * nx + col
                quads.append([idx, idx+1, idx+nx+1, idx+nx])
        cells = [("quad", np.array(quads, dtype=np.int64))]
        mesh = meshio.Mesh(points, cells, point_data={"u": U.flatten()})
        
        # Use to_absolute_path to ensure the directory is set as desired.
        output_dir = to_absolute_path("/workspace/my_examples/em_wave_2D/outputs/timestamp")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{output_prefix}_t{i}.vtk")
        mesh.write(filename)
        print(f"Exported {filename}")
