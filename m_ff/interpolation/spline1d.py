from __future__ import division, print_function

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


class Spline1DAngle(Spline1D):

    def __init__(self, x0, f):
        """

        :param x0: 1 dimensional array
        :param f: 1-dimensional array
        """

        super(Spline1D, self).__init__(x0, f)

    def ev_all(self, rs):
        # Energy and Forces as a function of a configuration
        ds = np.linalg.norm(rs, axis=1, keepdims=True)
        rs_hat = rs / ds

        fs_scalars, energy_single = super(Spline1D, self).ev_all(ds)

        tot_energy = np.sum(energy_single, axis=0)
        force_vectors = - np.sum(fs_scalars * rs_hat, axis=0)

        return tot_energy, force_vectors

    def ev_forces(self, rs):
        # Force as a function of a configuration
        ds = np.linalg.norm(rs, axis=1, keepdims=True)
        rs_hat = rs / ds

        fs_scalars = super(Spline1D, self).ev_forces(ds)

        force_vectors = - np.sum(fs_scalars * rs_hat, axis=0)

        return force_vectors

    def ev_energy(self, rs):
        # Energy as a function of a configuration
        ds = np.linalg.norm(rs, axis=1, keepdims=True)

        energy_single = super(Spline1D, self).ev_energy(ds)
        tot_energy = np.sum(energy_single, axis=0)

        return tot_energy

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        x_range, energies = data[:, 0], data[:, 1]

        return cls(x_range, energies)
