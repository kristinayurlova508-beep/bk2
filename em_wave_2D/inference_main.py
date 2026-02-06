# inference_main.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from omegaconf import DictConfig

@modulus.sym.main(config_path="config", config_name="config_el")
def inference_main(cfg: DictConfig) -> None:
    """
    Inference script for PINN.
    Uses the full config from file and extracts the architecture config.
    Evaluates the trained model on a grid over [0,0.5]x[0,0.5] at a fixed time,
    overlays the concrete object and antenna, and saves a PDF plot.
    """
    # Extract the minimal architecture configuration from the config file.
    arch_cfg = cfg.arch.fully_connected

    # Instantiate the network with the architecture config from file.
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=arch_cfg,
    )

    # Load the trained checkpoint (adjust the path as needed)
    checkpoint_path = "/workspace/my_examples/em_wave_2D/outputs/main/wave_network.0.pth"
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        wave_net.load_state_dict(state_dict)
        print("Loaded trained model from", checkpoint_path)
    except Exception as e:
        print("Could not load checkpoint:", e)

    # Set the model to evaluation mode
    wave_net.eval()

    # Define the domain grid: x, y in [0, 0.5]
    nx, ny = 200, 200
    x = np.linspace(0, 0.5, nx)
    y = np.linspace(0, 0.5, ny)
    X, Y = np.meshgrid(x, y)

    # Fix a time for the snapshot (e.g., t = 1e-9 s)
    t_fixed = 1e-9

    # Flatten the grid and prepare input array of shape (N, 3)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    T_flat = t_fixed * np.ones_like(X_flat)
    inputs = np.stack([X_flat, Y_flat, T_flat], axis=1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Create a dictionary mapping keys (as strings) to tensors.
    inputs_dict = {
        "x": inputs_tensor[:, 0:1],
        "y": inputs_tensor[:, 1:2],
        "t": inputs_tensor[:, 2:3],
    }

    # Evaluate the network using the dictionary of inputs.
    with torch.no_grad():
        outputs = wave_net(inputs_dict)
        print("Network outputs keys:", outputs.keys())
        # Use the correct key based on what is printed.
        if "u" in outputs:
            u_pred = outputs["u"].detach().cpu().numpy().flatten()
        elif Key("u") in outputs:
            u_pred = outputs[Key("u")].detach().cpu().numpy().flatten()
        else:
            raise KeyError("Output key 'u' not found in network outputs: " + str(outputs.keys()))

    # Reshape the output to the grid shape
    U = u_pred.reshape((ny, nx))

    # Create a plot of the wave field
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(U, extent=[0, 0.5, 0, 0.5], origin="lower", cmap="viridis")
    fig.colorbar(cax, label="Wave Field (u)")

    # Overlay the concrete object as a red rectangle.
    # Concrete object: from (0.15, 0.30) to (0.20, 0.35)
    rect = patches.Rectangle((0.15, 0.30), 0.05, 0.05,
                             linewidth=2, edgecolor='red', facecolor='none', label="Concrete Object")
    ax.add_patch(rect)

    # Overlay the antenna region as a blue circle.
    # Antenna: center (0.25, 0.45), radius 0.04
    circle = patches.Circle((0.25, 0.45), 0.04,
                            linewidth=2, edgecolor='blue', facecolor='none', label="Antenna")
    ax.add_patch(circle)

    ax.set_title(f"PINN Wave Field at t = {t_fixed:.1e} s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()

    # Save the plot as a PDF file
    pdf_filename = "pinn_wave_field.pdf"
    plt.savefig(pdf_filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    print("PDF saved as", pdf_filename)

if __name__ == "__main__":
    inference_main()
