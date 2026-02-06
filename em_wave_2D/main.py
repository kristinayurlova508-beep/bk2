# pinn_2d_single_region.py

import math
import os
import shutil
from sympy import Symbol
import modulus.sym
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import PointwiseInteriorConstraint, PointwiseBoundaryConstraint
from modulus.sym.key import Key
from modulus.sym.solver import Solver
from modulus.sym.hydra import instantiate_arch, ModulusConfig

# Import our custom modules
from plotter import WavePlotter
from pde_definitions import WaveEquationAtten, OpenBoundary, ImpedanceBC
from pde_no_source import WaveEquationAttenNoSource
from pde_source import WaveEquationAttenSource

def alpha_calc(f, mu, eps, sigma):
    """
    Computes the attenuation coefficient for a lossy medium using the formula:
    
        α = ω * sqrt( (μ * ε / 2) * ( sqrt(1 + (σ/(ωε))²) - 1 ) )
    
    Parameters:
        f     : frequency (Hz)
        mu    : permeability (H/m)
        eps   : permittivity (F/m)
        sigma : conductivity (S/m)
    
    Returns:
        alpha : attenuation coefficient (1/m)
    """
    omega = 2 * math.pi * f
    term = sigma / (omega * eps)
    return omega * math.sqrt((mu * eps / 2.0) * (math.sqrt(1 + term**2) - 1))

@modulus.sym.main(config_path="config", config_name="config_el")
def run(cfg: ModulusConfig) -> None:
    """
    Example PINN setup for a single 2D domain [0,0.5] x [0,0.5],
    with a 'concrete' object, an antenna source, and open boundaries.
    Frequency ~ 5 GHz, time in [0, 1e-8].
    """

    # ---------------------------------------------------------------
    # (Optional) Clear previous outputs
    # ---------------------------------------------------------------
    output_dir = "./outputs"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted old folder: {output_dir}")

    # ---------------------------------------------------------------
    # 1) Domain and geometry
    # ---------------------------------------------------------------
    # Single region: 0 <= x <= 0.5, 0 <= y <= 0.5
    rec = Rectangle((0, 0), (0.5, 0.5))

    # Example: place a 'concrete object' rectangle inside the domain
    # e.g., from x=0.15 to 0.20, y=0.30 to 0.35
    concrete_obj = Rectangle((0.15, 0.30), (0.20, 0.35))

    # Example: define an antenna region as a circle near the top
    # center ~ (0.25, 0.45), radius=0.04
    antenna_circle = Circle(center=(0.25, 0.45), radius=0.04)

    # ---------------------------------------------------------------
    # 2) Material properties & frequency
    # ---------------------------------------------------------------
    # Air properties
    mu_air = 4 * math.pi * 1e-7
    eps_air = 8.854e-12
    sigma_air = 1e-12  # low conductivity
    # Concrete properties
    mu_conc = mu_air
    eps_conc = 10 * 8.854e-12
    sigma_conc = 0.1

    f = 5e9  # 5 GHz

    # Calculate attenuation coefficients
    alpha_air = alpha_calc(f, mu_air, eps_air, sigma_air)
    alpha_conc = alpha_calc(f, mu_conc, eps_conc, sigma_conc)

    print(f"alpha_air = {alpha_air:.3e}, alpha_conc = {alpha_conc:.3e}")

    # Wave speeds
    c_air = 1.0 / math.sqrt(mu_air * eps_air)
    c_conc = 1.0 / math.sqrt(mu_conc * eps_conc)

    # ---------------------------------------------------------------
    # 3) PDE definitions
    # ---------------------------------------------------------------
    we_air = WaveEquationAttenNoSource(u="u", c=c_air, dim=2, time=True, alpha=alpha_air)
    we_conc = WaveEquationAttenNoSource(u="u", c=c_conc, dim=2, time=True, alpha=alpha_conc)
    we_source = WaveEquationAttenSource(
        x0=0.25,
        y0=0.45,
        sigma=0.04,
        amplitude=10.0,
        u="u",
        c=c_air,
        dim=2,
        time=True,
        alpha=alpha_air,
    )
    ob = OpenBoundary(u="u", c=c_air, dim=2, time=True)

    # ---------------------------------------------------------------
    # 4) Neural network
    # ---------------------------------------------------------------
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    # Combine PDE nodes
    nodes = (
        we_air.make_nodes()
        + we_conc.make_nodes()
        + we_source.make_nodes()
        + ob.make_nodes()
        + [wave_net.make_node(name="wave_network")]
    )

    # ---------------------------------------------------------------
    # 5) Build the domain and add constraints
    # ---------------------------------------------------------------
    domain = Domain()

    # Time range: t from 0 to 1e-8 seconds
    t_sym = Symbol("t")
    time_range = {t_sym: (0, 1e-8)}

    interior_air = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec - concrete_obj,
        outvar={"wave_equation_no_source": 0},
        batch_size=256,
        parameterization=time_range,
    )
    domain.add_constraint(interior_air, "InteriorAir")

    interior_concrete = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=concrete_obj,
        outvar={"wave_equation_no_source": 0},
        batch_size=256,
        parameterization=time_range,
    )
    domain.add_constraint(interior_concrete, "InteriorConcrete")

    interior_antenna = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=antenna_circle,
        outvar={"wave_equation_with_source": 0},
        batch_size=256,
        parameterization=time_range,
    )
    domain.add_constraint(interior_antenna, "AntennaSource")

    boundary_ob = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"open_boundary": 0},
        batch_size=256,
        parameterization=time_range,
    )
    domain.add_constraint(boundary_ob, "OpenBoundary")

    # ---------------------------------------------------------------
    # 6) Solve
    # ---------------------------------------------------------------
    # Remove the extra nodes argument for the current API
    slv = Solver(cfg, domain)
    if hasattr(slv, "_trainer"):
        for group in slv._trainer.optimizer.param_groups:
            group["foreach"] = False
    slv.solve()

    print("PINN training finished.")

    # ---------------------------------------------------------------
    # 7) (Optional) Post-processing / Inference
    # ---------------------------------------------------------------
    # You can call an inference script to generate plots, e.g.:
    # os.system("python inference_main.py")

if __name__ == "__main__":
    run()
