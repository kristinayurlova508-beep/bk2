# plotter.py
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io.plotter import ValidatorPlotter

class WavePlotter(ValidatorPlotter):
    """Custom validator plotter that only plots x,y dimensions."""

    def __call__(self, invar, true_outvar, pred_outvar):
        # Only use x and y for plotting
        invar = {k: v for k, v in invar.items() if k in ["x", "y"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs
