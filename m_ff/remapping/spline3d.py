from __future__ import division, print_function
from itertools import combinations
import numpy as np

import _tricub


class BaseSpline3D(object):

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


class Spline3DAngle(BaseSpline3D):

    def ev_energy(self, rs, **kwargs):
        # Vectorize
        r1, r2, r1_hat, r2_hat, cosphi = self.vectorize(rs)

        # Interpolation
        val = super().ev_energy(r1, r2, np.arccos(cosphi))

        return np.sum(val, axis=0)

    def ev_energy_fast(self, rs, **kwargs):
        # Vectorize
        r1, r2, r1_hat, r2_hat, cosphi = self.vectorize(rs)

        # Interpolation
        val = super().ev_energy_fast(r1, r2, np.arccos(cosphi))

        return np.sum(val, axis=0)

    def ev_forces(self, rs, **kwargs):
        # Vectorize
        r1, r2, r1_hat, r2_hat, cosphi = self.vectorize(rs)

        # Interpolation
        de_dr1, de_dr2, de_dphi = super().ev_forces(r1, r2, np.arccos(cosphi))

        dphi_dr0 = - 1. / (np.sqrt(1 - cosphi ** 2)) * (
                (r1_hat / r2 + r2_hat / r1) - cosphi * (r1_hat / r1 + r2_hat / r2))

        tri_force = -(de_dr1 * r1_hat + de_dr2 * r2_hat + de_dphi * dphi_dr0)

        tri_force = np.sum(tri_force, axis=0)

        return tri_force

    def ev_forces_fast(self, rs, **kwargs):
        # Vectorize
        r1, r2, r1_hat, r2_hat, cosphi = self.vectorize(rs)

        # Interpolation
        de_dr1, de_dr2, de_dphi = super().ev_forces_fast(r1, r2, np.arccos(cosphi))

        dphi_dr0 = - 1. / (np.sqrt(1 - cosphi ** 2)) * (
                (r1_hat / r2 + r2_hat / r1) - cosphi * (r1_hat / r1 + r2_hat / r2))

        tri_force = -(de_dr1 * r1_hat + de_dr2 * r2_hat )

        tri_force = np.sum(tri_force, axis=0)

        return tri_force

    def ev_all(self, rs, **kwargs):
        # Vectorize
        r1, r2, r1_hat, r2_hat, cosphi = self.vectorize(rs)

        # Interpolation
        val, de_dr1, de_dr2, de_dphi = super().ev_all(r1, r2, np.arccos(cosphi))

        dphi_dr0 = - 1. / (np.sqrt(1 - cosphi ** 2)) * (
                (r1_hat / r2 + r2_hat / r1) - cosphi * (r1_hat / r1 + r2_hat / r2))

        tri_force = -(de_dr1 * r1_hat + de_dr2 * r2_hat + de_dphi * dphi_dr0)

        tri_force = np.sum(tri_force, axis=0)

        return np.sum(val, axis=0), tri_force

    def ev_all_fast(self, rs, **kwargs):
        # Vectorize
        r1, r2, r1_hat, r2_hat, cosphi = self.vectorize(rs)

        # Interpolation
        val, de_dr1, de_dr2, de_dphi = super().ev_all_fast(r1, r2, np.arccos(cosphi))

        dphi_dr0 = - 1. / (np.sqrt(1 - cosphi ** 2)) * (
                (r1_hat / r2 + r2_hat / r1) - cosphi * (r1_hat / r1 + r2_hat / r2))

        tri_force = -(de_dr1 * r1_hat + de_dr2 * r2_hat + de_dphi * dphi_dr0)

        tri_force = np.sum(tri_force, axis=0)

        return np.sum(val, axis=0), tri_force

    @staticmethod
    def vectorize(rs):
        # Calculate the norm and the normal vectors
        ds = np.linalg.norm(rs, axis=1, keepdims=True)
        # ds = np.sqrt(np.einsum('nd, nd -> n', rs, rs))

        rs_hat = rs / ds
        # rs_hat = np.einsum('nd, n -> nd', rs, 1. / ds)

        r1, r2, r1_hat, r2_hat = Spline3D.combinations(ds, rs_hat)

        cosphi = np.clip(np.einsum('nd, nd -> n', r1_hat, r2_hat), -1, 1).reshape(-1, 1)

        return r1, r2, r1_hat, r2_hat, cosphi

    @staticmethod
    def combinations(ds, rs_hat):
        r1r2 = np.array(list(combinations(ds, 2)))
        r1 = r1r2[:, 0]
        r2 = r1r2[:, 1]

        r1_r2_hat = np.array(list(combinations(rs_hat, 2)))
        r1_hat = r1_r2_hat[:, 0, :]
        r2_hat = r1_r2_hat[:, 1, :]

        # n, m = rs_hat.shape
        # dtype = rs_hat.dtype
        #
        # # number of outputs
        # n_out = n * (n - 1) // 2
        #
        # # Allocate arrays
        # r1 = np.zeros([n_out, 1], dtype=dtype)
        # r2 = np.zeros([n_out, 1], dtype=dtype)
        #
        # r1_hat = np.zeros([n_out, m], dtype=dtype)
        # r2_hat = np.zeros([n_out, m], dtype=dtype)
        #
        # for i, ((r1_i, r1_hat_i), (r2_i, r2_hat_i)) in enumerate(combinations(zip(ds, rs_hat), r=2)):
        #     r1[i], r2[i] = r1_i, r2_i
        #     r1_hat[i, :], r2_hat[i, :] = r1_hat_i, r2_hat_i

        return r1, r2, r1_hat, r2_hat

    @classmethod
    def from_file(cls, filename):
        grid_energies = np.load(filename)
        three_energies = grid_energies[:, :, :, 2]
        rgrid = grid_energies[:, 0, 0, 0]
        phigrid = grid_energies[0, 0, :, 1]

        return cls(rgrid, rgrid, phigrid, three_energies)


if __name__ == '__main__':
    a = _tricub.reg_ev_energy()
    pass
