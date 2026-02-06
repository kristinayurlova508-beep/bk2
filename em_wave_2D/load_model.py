import os
import torch
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.key import Key

def load_my_trained_model_somehow(cfg: ModulusConfig, device):
    """
    Instantiate the network architecture from the config, load its checkpoint,
    move it to the specified device, and return the inference node.
    """
    # Instantiate the network architecture using the configuration.
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("ux"), Key("uy")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Define the checkpoint path. Adjust as needed.
    checkpoint_path = os.path.join("/workspace/my_examples/em_wave_2D/outputs/main/wave_network.0.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # Load the state dictionary into the network.
    wave_net.load_state_dict(state_dict)
    
    # Move the entire network to the desired device.
    wave_net = wave_net.to(device)
    
    # Now create the network node.
    wave_network_node = wave_net.make_node(name="wave_network")
    # (If necessary, try to move the underlying module as well.)
    print(f"Loaded trained network from {checkpoint_path} onto {device}")
    return wave_network_node
