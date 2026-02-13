import modulus.sym
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.key import Key
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.solver import Solver
from modulus.sym.domain.constraint import PointwiseInteriorConstraint
from sympy import symbols, sin, exp, sqrt, pi, Piecewise

print("=== glass.py START ===", flush=True)
import sys
print("ARGS:", sys.argv, flush=True)


@modulus.sym.main(config_path="config", config_name="config")
def run(cfg) -> None:
    # -------------------------
    # Definovanie symbolov
    # -------------------------
    x, y = symbols("x y", real=True)

    # -------------------------
    # Parametre
    # -------------------------
    # Prvé prostredie: vzduch (nízka absorpcia)
    absorption_air = 0.1
    # Druhé prostredie: glass (absorpcia podľa materiálu)
    absorption_mat = 0.3

    amplitude = 1.0
    lambda_val = 0.3

    # Vertikálna hranica medzi prostredím – x < boundary_x: vzduch, x >= boundary_x: glass
    boundary_x = 1.5

    # Koeficienty pre odraz a transmisiu na rozhraní
    R = 0.2   # odraz
    T = 0.8   # transmisia

    # Pozícia antény
    x0, y0 = 1, 1

    # -------------------------
    # Vzdušná oblasť (x < boundary_x)
    # -------------------------
    r_air = sqrt((x - x0)**2 + (y - y0)**2)
    u_air_direct = amplitude * sin(2*pi*r_air/lambda_val) * exp(-absorption_air*r_air)

    # Odrazená vlna: zrkadlový obraz antény cez hranicu x = boundary_x
    x_ref = 2*boundary_x - x0
    y_ref = y0
    r_ref = sqrt((x - x_ref)**2 + (y - y_ref)**2)
    u_air_reflected = -amplitude * R * sin(2*pi*r_ref/lambda_val) * exp(-absorption_air*r_ref)

    u_expr_air = u_air_direct + u_air_reflected

    # -------------------------
    # Oblasť glass (x >= boundary_x)
    # -------------------------
    eps = 1e-12  # na zabránenie delenia nulou
    slope = (y - y0) / ((x - x0) + eps)
    y_int = y0 + slope*(boundary_x - x0)

    # r_in: vzdialenosť od antény k prieniku hranice
    r_in = sqrt((boundary_x - x0)**2 + (y_int - y0)**2)
    # r_out: vzdialenosť od prieniku k bodu (x,y)
    r_out = sqrt((x - boundary_x)**2 + (y - y_int)**2)
    total_path = r_in + r_out

    # Transmisný člen: vlna, ktorá prejde hranicou (s tlmením v materiáli)
    u_transmitted = amplitude * T * sin(2*pi*total_path/lambda_val) * exp(-absorption_mat*total_path)

    # Scattering (rozptyl) – amplitúda rozptylu závisí od absorpcie materiálu
    amplitude_scattered = amplitude * (absorption_mat / (absorption_mat + 1))
    u_scattered = amplitude_scattered * sin(2*pi*r_out/lambda_val) * exp(-absorption_mat*r_out)

    u_expr_mat = u_transmitted + u_scattered

    # -------------------------
    # Kombinácia cez Piecewise
    # -------------------------
    u_expr = Piecewise(
        (u_expr_air, x < boundary_x),
        (u_expr_mat, True)
    )

    # -------------------------
    # Nastavenie PINN
    # -------------------------
    geom = Rectangle((0, 0), (3, 2))
    input_keys = [Key("x"), Key("y")]
    output_keys = [Key("u")]

    net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
    nodes = [net.make_node("wave")]

    wave_constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"u": u_expr},
        batch_size=cfg.batch_size.Interior
    )

    domain = Domain()
    domain.add_constraint(wave_constraint, "wave_constraint")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
