# pde_definitions.py
import math
from sympy import Symbol, Function, Number, sqrt, sin, exp

from modulus.sym.eq.pde import PDE

class WaveEquationAtten(PDE):
    """
    Wave equation with attenuation.

    Parameters:
      u : str
          Dependent variable.
      c : float, Sympy Symbol/Expr, or str
          Wave speed coefficient.
      dim : int
          Dimension of the problem.
      time : bool
          If time-dependent or not.
      mixed_form : bool
          If True, use mixed formulation.
      alpha : float
          Attenuation coefficient.
    """
    name = "WaveEquationAtten"

    def __init__(self, u="u", c="c", dim=3, time=True, mixed_form=False, alpha=0.1):
        self.u = u
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        self.alpha = alpha

        # coordinates and time symbol
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Define source term with improved stability
        def source_term(inputs):
            x_loc = inputs["x"]
            y_loc = inputs["y"]
            t_loc = inputs["t"]
            x0 = 0.5
            y0 = 0.5
            sigma_loc = 0.1  # Smoother source
            scaling = 0.1
            return scaling * sin(t_loc) * exp(-(((x_loc - x0)**2) + ((y_loc - y0)**2)) / (2 * sigma_loc**2))
        
        # Convert u and c to appropriate sympy expressions
        u_expr = Function(u)(*input_variables)
        if isinstance(c, str):
            c_expr = Function(c)(*input_variables)
        elif isinstance(c, (float, int)):
            c_expr = Number(c)
        
        self.equations = {}
        if not self.mixed_form:
            self.equations["wave_equation"] = (
                u_expr.diff(t, 2)
                - c_expr**2 * u_expr.diff(x, 2)
                - c_expr**2 * u_expr.diff(y, 2)
                - c_expr**2 * u_expr.diff(z, 2)
                - 2 * self.alpha * u_expr.diff(t)
                - source_term(input_variables)
            )
        else:
            # Mixed form formulation (not detailed here)
            raise NotImplementedError("Mixed form not implemented in this refactoring.")

class OpenBoundary(PDE):
    """
    Open boundary condition for wave problems.
    """
    name = "OpenBoundary"

    def __init__(self, u="u", c="c", dim=2, time=True):
        self.u = u
        self.dim = dim
        self.time = time

        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")
        t = Symbol("t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        u_expr = Function(u)(*input_variables)
        if isinstance(c, str):
            c_expr = Function(c)(*input_variables)
        elif isinstance(c, (float, int)):
            c_expr = Number(c)

        self.equations = {}
        self.equations["open_boundary"] = (
            u_expr.diff(t)
            + normal_x * c_expr * u_expr.diff(x)
            + normal_y * c_expr * u_expr.diff(y)
            + normal_z * c_expr * u_expr.diff(z)
        )

class ImpedanceBC(PDE):
    """
    Impedance Boundary Condition for electromagnetic waves.
    """
    name = "ImpedanceBC"

    def __init__(self, ux="ux", uy="uy", uz="uz", c="c", mu1=1, mu2=1, epsilon1=1, epsilon2=1, dimension=2, time=True):
        assert dimension in (2, 3), "dimension must be 2 or 3"
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.dim = dimension
        self.time = time

        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        ux_expr = Function(ux)(*input_variables)
        uy_expr = Function(uy)(*input_variables)
        uz_expr = Function(uz)(*input_variables)

        z1 = sqrt(mu1 / epsilon1)
        z2 = sqrt(mu2 / epsilon2)

        normal_x, normal_y, normal_z = Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")
        n = [normal_x, normal_y]
        if self.dim == 3:
            n.append(normal_z)
        
        # Incident field components (assume they are given by the parameter names)
        E_incident = [self.ux, self.uy]
        if self.dim == 3:
            E_incident.append(self.uz)
        
        # Compute the parallel (simplistic approach) and perpendicular components
        E_parallel = [Symbol(v) * n_i for v, n_i in zip(E_incident, n)]
        E_perpendicular = [Symbol(E_incident[i]) - E_parallel[i] for i in range(self.dim)]
        A_r = (z1 - z2) / (z1 + z2)
        A_t = (2 * z1) / (z1 + z2)
        E_reflected = [A_r * E_perpendicular[i] for i in range(self.dim)]
        E_transmitted = [A_t * E_perpendicular[i] for i in range(self.dim)]

        self.equations = {}
        for i, comp in enumerate(['x', 'y', 'z'][:self.dim]):
            self.equations[f"ImpedanceBC_reflected_{comp}"] = E_reflected[i]
            self.equations[f"ImpedanceBC_transmitted_{comp}"] = E_transmitted[i]
