import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Spline1D(InterpolatedUnivariateSpline):

    def __init__(self, x0, f):
        """

        :param x0: 1 dimensional array
        :param f: 1-dimensional array
        """

        super(Spline1D, self).__init__(x0, f, k=3, ext=3)

    def ev_all(self, x):
        return self.ev_energy(x), self.ev_forces(x)

    def ev_forces(self, x):
        fs_scalars = super(Spline1D, self).__call__(x, nu=1)
        return fs_scalars

    def ev_energy(self, x):
        energy_single = super(Spline1D, self).__call__(x, nu=0)
        return energy_single

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        x_range, energies = data[:, 0], data[:, 1]

        return cls(x_range, energies)
