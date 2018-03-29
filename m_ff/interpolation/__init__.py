import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
# noinspection PyUnresolvedReferences
from . import _tricub


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


class Spline3D(object):

    def __init__(self, x, y, z, f):

        assert f.shape == (x.size, y.size, z.size), 'dimensions do not match f'
        assert np.all(np.diff(x) > 0) & np.all(np.diff(y) > 0) & np.all(np.diff(z) > 0), \
            'x, y or z is not monotonically increasing'

        self._xlim = np.array([x.min(), x.max()])
        self._ylim = np.array([y.min(), y.max()])
        self._zlim = np.array([z.min(), z.max()])

        self._x = np.pad(x, pad_width=1, mode='constant', constant_values=(2 * x[0] - x[1], 2 * x[-1] - x[-2]))
        self._y = np.pad(y, pad_width=1, mode='constant', constant_values=(2 * y[0] - y[1], 2 * y[-1] - y[-2]))
        self._z = np.pad(z, pad_width=1, mode='constant', constant_values=(2 * z[0] - z[1], 2 * z[-1] - z[-2]))

        # boundary = 'clamped'
        self._f = np.pad(f, pad_width=1, mode='edge')

    def _check_bounds(self, x_new, y_new, z_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Args:
            x_new (float array):

            y_new (float array):

        Returns:
            out_of_bounds (Boolean array): The mask on x_new and y_new of
            values that are NOT of bounds.
        """

        below_bounds_x = x_new < self._xlim[0]
        above_bounds_x = x_new > self._xlim[1]

        below_bounds_y = y_new < self._ylim[0]
        above_bounds_y = y_new > self._ylim[1]

        below_bounds_z = z_new < self._zlim[0]
        above_bounds_z = z_new > self._zlim[1]

        # !! Could provide more information about which values are out of bounds
        if np.any(below_bounds_x):
            raise ValueError('A value in x is below the interpolation range.')
        if np.any(above_bounds_x):
            raise ValueError('A value in x is above the interpolation range.')
        if np.any(below_bounds_y):
            raise ValueError('A value in y is below the interpolation range.')
        if np.any(above_bounds_y):
            raise ValueError('A value in y is above the interpolation range.')
        if np.any(below_bounds_z):
            raise ValueError('A value in z is below the interpolation range.')
        if np.any(above_bounds_z):
            raise ValueError('A value in z is above the interpolation range.')

    def ev_energy_fast(self, x, y, z):

        val = _tricub.reg_ev_energy(z, y, x, self._f, self._z, self._y, self._x)

        return val[:, np.newaxis]

    def ev_energy(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_energy_fast(x, y, z)

    def ev_forces_fast(self, x, y, z):

        val_dx2, val_dx1, val_dx0 = _tricub.reg_ev_forces(z, y, x, self._f, self._z, self._y, self._x)

        return val_dx0[:, np.newaxis], val_dx1[:, np.newaxis], val_dx2[:, np.newaxis]

    def ev_forces(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_forces_fast(x, y, z)

    def ev_all_fast(self, x, y, z):

        val, val_dx2, val_dx1, val_dx0 = _tricub.reg_ev_all(z, y, x, self._f, self._z, self._y, self._x)

        return val[:, np.newaxis], val_dx0[:, np.newaxis], val_dx1[:, np.newaxis], val_dx2[:, np.newaxis]

    def ev_all(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_all_fast(x, y, z)
