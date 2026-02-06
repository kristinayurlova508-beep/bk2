# pde_no_source.py

from sympy import Symbol, Function, Number
from modulus.sym.eq.pde import PDE

class WaveEquationAttenNoSource(PDE):
    """
    Wave equation with attenuation, no source term.
    wave_equation_no_source = u_tt - c^2 ∇²u - 2 alpha u_t = 0
    """
    name = "WaveEquationNoSource"

    def __init__(self, u="u", c="c", dim=2, time=True, alpha=0.1):
        # define coordinates, etc.
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")
        input_vars = {"x": x, "y": y, "z": z, "t": t}
        if dim == 2:
            input_vars.pop("z")
        if not time:
            input_vars.pop("t")

        u_expr = Function(u)(*input_vars)
        if isinstance(c, (float, int)):
            from sympy import Number
            c_expr = Number(c)
        else:
            # c is symbolic or a function
            c_expr = Function(c)(*input_vars)

        self.equations = {}
        self.equations["wave_equation_no_source"] = (
            u_expr.diff(t, 2)
            - c_expr**2 * (u_expr.diff(x, 2) + u_expr.diff(y, 2))
            - 2*alpha*u_expr.diff(t)
        )
