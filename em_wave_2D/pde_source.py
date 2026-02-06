# pde_source.py
from sympy import Symbol, Function, Number, sin, exp
from modulus.sym.eq.pde import PDE

class WaveEquationAttenSource(PDE):
    """
    Wave equation with attenuation AND a source term.
    
    wave_equation_with_source = u_tt - c^2 ∇²u - 2α u_t - source(x,y,t) = 0
    
    The antenna parameters (x0, y0, sigma, amplitude) must be provided.
    """
    name = "WaveEquationWithSource"

    def __init__(self, x0, y0, sigma, amplitude, u="u", c="c", dim=2, time=True, alpha=0.1):
        # Define coordinates and time
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
            c_expr = Function(c)(*input_vars)
        
        # Use passed antenna parameters for the source term:
        source_expr = amplitude * sin(t) * exp(
            -(((x - x0)**2) + ((y - y0)**2)) / (2 * sigma**2)
        )
        
        self.equations = {}
        self.equations["wave_equation_with_source"] = (
            u_expr.diff(t, 2)
            - c_expr**2 * (u_expr.diff(x, 2) + u_expr.diff(y, 2))
            - 2 * alpha * u_expr.diff(t)
            - source_expr
        )
