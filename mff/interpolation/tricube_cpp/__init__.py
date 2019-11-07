import numpy as np

from mff.interpolation.tricube_cpp import _tricube


class Spline3D(object):

    def __init__(self, x, y, z, f):

        assert f.shape == (x.size, y.size, z.size), "dimensions do not match f"
        assert np.all(np.diff(x) > 0) & np.all(np.diff(y) > 0) & np.all(np.diff(z) > 0), \
            "x, y or z is not monotonically increasing"

        self._xlim = np.array([x.min(), x.max()])
        self._ylim = np.array([y.min(), y.max()])
        self._zlim = np.array([z.min(), z.max()])

        self._x = np.pad(x, pad_width=1, mode='constant',
                         constant_values=(2 * x[0] - x[1], 2 * x[-1] - x[-2]))
        self._y = np.pad(y, pad_width=1, mode='constant',
                         constant_values=(2 * y[0] - y[1], 2 * y[-1] - y[-2]))
        self._z = np.pad(z, pad_width=1, mode='constant',
                         constant_values=(2 * z[0] - z[1], 2 * z[-1] - z[-2]))

        boundary = 'natural'
        # self._f = np.pad(f, pad_width=1, mode='edge')

        self._f = np.zeros(np.array(f.shape) + (2, 2, 2))
        # place f in center, so that it is padded by unfilled values on all sides
        self._f[1:-1, 1:-1, 1:-1] = f
        if boundary == 'clamped':
            # faces
            self._f[(0, -1), 1:-1, 1:-1] = f[(0, -1), :, :]
            self._f[1:-1, (0, -1), 1:-1] = f[:, (0, -1), :]
            self._f[1:-1, 1:-1, (0, -1)] = f[:, :, (0, -1)]
            # verticies
            self._f[(0, 0, -1, -1), (0, -1, 0, -1), 1:-
                    1] = f[(0, 0, -1, -1), (0, -1, 0, -1), :]
            self._f[(0, 0, -1, -1), 1:-1, (0, -1, 0, -1)
                    ] = f[(0, 0, -1, -1), :, (0, -1, 0, -1)]
            self._f[1:-1, (0, 0, -1, -1), (0, -1, 0, -1)] = f[:,
                                                              (0, 0, -1, -1), (0, -1, 0, -1)]
            # corners
            self._f[(0, 0, 0, 0, -1, -1, -1, -1), (0, 0, -1, -1, 0, 0, -1, -1), (0, -1, 0, -1, 0, -1, 0, -1)] = \
                f[(0, 0, 0, 0, -1, -1, -1, -1), (0, 0, -1, -1,
                                                 0, 0, -1, -1), (0, -1, 0, -1, 0, -1, 0, -1)]
        elif boundary == 'natural':
            # faces
            self._f[(0, -1), 1:-1, 1:-1] = 2 * \
                f[(0, -1), :, :] - f[(1, -2), :, :]
            self._f[1:-1, (0, -1), 1:-1] = 2 * \
                f[:, (0, -1), :] - f[:, (1, -2), :]
            self._f[1:-1, 1:-1, (0, -1)] = 2 * \
                f[:, :, (0, -1)] - f[:, :, (1, -2)]
            # verticies
            self._f[(0, 0, -1, -1), (0, -1, 0, -1), 1:-1] = \
                4 * f[(0, 0, -1, -1), (0, -1, 0, -1), :] - \
                f[(1, 1, -2, -2), (0, -1, 0, -1), :] - \
                f[(0, 0, -1, -1), (1, -2, 1, -2), :] - \
                f[(1, 1, -2, -2), (1, -2, 1, -2), :]
            self._f[(0, 0, -1, -1), 1:-1, (0, -1, 0, -1)] = \
                4 * f[(0, 0, -1, -1), :, (0, -1, 0, -1)] - \
                f[(1, 1, -2, -2), :, (0, -1, 0, -1)] - \
                f[(0, 0, -1, -1), :, (1, -2, 1, -2)] - \
                f[(1, 1, -2, -2), :, (1, -2, 1, -2)]
            self._f[1:-1, (0, 0, -1, -1), (0, -1, 0, -1)] = \
                4 * f[:, (0, 0, -1, -1), (0, -1, 0, -1)] - \
                f[:, (1, 1, -2, -2), (0, -1, 0, -1)] - \
                f[:, (0, 0, -1, -1), (1, -2, 1, -2)] - \
                f[:, (1, 1, -2, -2), (1, -2, 1, -2)]
            # corners
            self._f[(0, 0, 0, 0, -1, -1, -1, -1), (0, 0, -1, -1, 0, 0, -1, -1), (0, -1, 0, -1, 0, -1, 0, -1)] = \
                8 * f[(0, 0, 0, 0, -1, -1, -1, -1), (0, 0, -1, -1, 0, 0, -1, -1), (0, -1, 0, -1, 0, -1, 0, -1)] - \
                f[(1, 1, 1, 1, -2, -2, -2, -2), (0, 0, -1, -1, 0, 0, -1, -1), (0, -1, 0, -1, 0, -1, 0, -1)] - \
                f[(0, 0, 0, 0, -1, -1, -1, -1), (1, 1, -2, -2, 1, 1, -2, -2), (0, -1, 0, -1, 0, -1, 0, -1)] - \
                f[(0, 0, 0, 0, -1, -1, -1, -1), (0, 0, -1, -1, 0, 0, -1, -1), (1, -2, 1, -2, 1, -2, 1, -2)] - \
                f[(1, 1, 1, 1, -2, -2, -2, -2), (1, 1, -2, -2, 1, 1, -2, -2), (0, -1, 0, -1, 0, -1, 0, -1)] - \
                f[(0, 0, 0, 0, -1, -1, -1, -1), (1, 1, -2, -2, 1, 1, -2, -2), (1, -2, 1, -2, 1, -2, 1, -2)] - \
                f[(1, 1, 1, 1, -2, -2, -2, -2), (0, 0, -1, -1, 0, 0, -1, -1), (1, -2, 1, -2, 1, -2, 1, -2)] - \
                f[(1, 1, 1, 1, -2, -2, -2, -2), (1, 1, -2, -2,
                                                 1, 1, -2, -2), (1, -2, 1, -2, 1, -2, 1, -2)]

    @property
    def data(self):
        return self._f[1:-1, 1:-1, 1:-1]

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

        val = _tricube.reg_ev_energy(
            x, y, z, self._f, self._x, self._y, self._z)

        return val[:, np.newaxis]

    def ev_energy(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_energy_fast(x, y, z)

    def ev_forces_fast(self, x, y, z):

        val_dx0, val_dx1, val_dx2 = _tricube.reg_ev_forces(
            x, y, z, self._f, self._x, self._y, self._z)

        return val_dx0[:, np.newaxis], val_dx1[:, np.newaxis], val_dx2[:, np.newaxis]

    def ev_forces(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_forces_fast(x, y, z)

    def ev_all_fast(self, x, y, z):

        val, val_dx0, val_dx1, val_dx2 = _tricube.reg_ev_all(
            x, y, z, self._f, self._x, self._y, self._z)

        return val[:, np.newaxis], val_dx0[:, np.newaxis], val_dx1[:, np.newaxis], val_dx2[:, np.newaxis]

    def ev_all(self, xi, yi, zi):

        x = np.atleast_1d(xi)
        y = np.atleast_1d(yi)
        z = np.atleast_1d(zi)  # This will not modify x1,y1,z1.

        self._check_bounds(x, y, z)

        return self.ev_all_fast(x, y, z)

    def __call__(self, xi, yi, zi, nu=0):
        if nu == 0:
            return self.ev_energy(xi, yi, zi)
        elif nu == 1:
            return self.ev_forces(xi, yi, zi)
        else:
            raise NotImplementedError()

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        x, y, z, energies = data['x'], data['y'], data['z'], data['f']

        return cls(x, y, z, energies)

    def save(self, filename):

        np.savez_compressed(
            filename, x=self._x[1:-1], y=self._y[1:-1], z=self._z[1:-1], f=self._f[1:-1, 1:-1, 1:-1])


if __name__ == '__main__':
    from pathlib import Path

    n = 10
    x, y, z = np.linspace(0, 10, n), np.linspace(
        0, 10, n), np.linspace(0, 10, n)

    f = np.random.rand(n, n, n)
    g1 = Spline3D(x, y, z, f)

    filename = Path('grid.npz')

    g1.save(filename)

    g2 = Spline3D.load(filename)

    print(g1.ev_all(3., 4.5, 6.))
    print(g2.ev_all(3., 4.5, 6.))

    # x, y, z = np.linspace(0, 2, 3), np.linspace(0, 2, 3), np.linspace(0, 2, 3)
    # f = np.linspace(0, 3 ** 3 - 1, 3 ** 3).reshape(3, 3, 3)
    #
    # print(x, y, z)
    # print(f)
    #
    # s = Spline3D(x, y, z, f)
    #
    # print(f[0, 0, 0], s.ev_energy(0, 0, 0))
    # print(f[0, 0, 1], s.ev_energy(0, 0, 1))
    #
    # print(s.ev_energy(0, 0, 0.5))

    # x, y, z = np.linspace(1, 2, 3), np.linspace(1, 3, 3), np.linspace(1, 4, 4)
    # print(x,y,z)
    # f = x[:, np.newaxis, np.newaxis] * y[np.newaxis, :, np.newaxis]**2 * z[np.newaxis, np.newaxis, :]**3
    # print(f.shape)
    # print(type(f))
    # print(f)
    #
    # s = Spline3D(x, y, z, f)
    # for xi in range(len(x)):
    #     for yi in range(len(y)):
    #         for zi in range(len(z)):
    #
    #             print(xi, yi, zi, f[xi, yi, zi], s.ev_energy([x[xi]], [y[yi]], [z[zi]]))
    #
    #
    # print((0*z+x[1])*(0*z+y[1])**2 * z**3)
    # print(s.ev_energy(0*z+x[1], 0*z+y[1], z))
    #
    # x, y, z = np.linspace(1, 4, 4), np.linspace(1, 5, 5), np.linspace(1, 6, 6)
    # print(x, y, z)
    # f = x[:, np.newaxis, np.newaxis] + y[np.newaxis, :, np.newaxis] ** 2 + z[np.newaxis, np.newaxis, :] ** 2
    #
    # s = Spline3D(x, y, z, f)
    #
    # for xi in range(len(x)):
    #     for yi in range(len(y)):
    #         for zi in range(len(z)):
    #             dx = 1
    #             dy = 2 * y[yi]
    #             dz = 2 * z[zi]
    #             print(xi, yi, zi, f[xi, yi, zi], dx, dy, dz, *s.ev_all(x[xi], y[yi], z[zi]))
