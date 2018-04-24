import numpy as np
from m_ff.interpolation.tricube_cython.consts import A


class Spline3D(object):
    def __init__(self, x, y, z, f):

        assert f.shape == (x.size, y.size, z.size), "dimensions do not match f"
        assert np.all(np.diff(x) > 0) & np.all(np.diff(y) > 0) & np.all(np.diff(z) > 0), \
            "x, y or z is not monotonically increasing"

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

        for i in range(len(x)):
            pass

        val = x

        return val[:, np.newaxis]

    def voxel(self):
        pass

    def tricubic_get_coeff_stacked(self):
        pass

    def tricubic_eval(self):
        pass

    def ev_energy(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_energy_fast(x, y, z)


if __name__ == '__main__':
    x, y, z = np.linspace(0, 2, 3), np.linspace(0, 2, 3), np.linspace(0, 2, 3)
    f = np.linspace(0, 3 ** 3 - 1, 3 ** 3).reshape(3, 3, 3)

    print(x, y, z)
    print(f)

    s = Spline3D(x, y, z, f)

    print(f[0, 0, 0], s.ev_energy(0, 0, 0))
    print(f[0, 0, 1], s.ev_energy(0, 0, 1))

    print(s.ev_energy(0, 0, 0.5))
