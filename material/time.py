import modulus.sym
from modulus.sym.hydra import ModulusConfig
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.key import Key
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.solver import Solver
from modulus.sym.domain.constraint import PointwiseInteriorConstraint, PointwiseBoundaryConstraint
from sympy import symbols, sin, exp, sqrt, pi
 
@modulus.sym.main(config_path="config", config_name="config")
def run(cfg: ModulusConfig) -> None:
     
     # -------------------------
     # Define the analytic wave function
     # -------------------------
     # Symbolic variables for spatial coordinates
     x, y = symbols("x y")
 
     # Define the wave function: u(x,y) = sin(2*pi*r/λ) * exp(-3*r), where r = sqrt(x^2 + y^2)
     absorption_air = 1.5
     amplitude = 1.0
     lambda_val = 0.3

     x0, y0 = 1,1
     r_air = sqrt((x - x0)**2 + (y - y0)**2)
     u_expr = amplitude * sin(2*pi*r_air/lambda_val) * exp(-absorption_air*r_air)
 
     # -------------------------
     # Set up the geometry and PINN network
     # -------------------------
     # Create a 2D rectangular geometry over the domain [-1, 1] x [-1, 1]
     geom = Rectangle((0, 0), (2, 2))
 
     # Define input and output keys for the network
     input_keys = [Key("x"), Key("y")]
     output_keys = [Key("u")]
 
     # Build a fully connected neural network with three hidden layers of 50 neurons each
     net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
 
     nodes = [net.make_node("wave")]
 
     # -------------------------
     # Define the pointwise constraint that forces the network to learn u_expr
     # -------------------------
     wave_constraint = PointwiseInteriorConstraint(
         nodes=nodes,
         geometry=geom,
         outvar={"u": u_expr},
         batch_size=cfg.batch_size.Interior  # Adjust batch size as needed
     )
 
     # Create a domain and add the constraint
     domain = Domain()
     domain.add_constraint(wave_constraint, "wave_constraint")
 
     # -------------------------
     # Configure and run the solver
     # -------------------------
     slv = Solver(cfg, domain)
     slv.solve()
 
     
     
 
if __name__ == "__main__":
     # Create and run the solver using the defined domain and configuration
     run()