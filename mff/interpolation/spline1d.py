import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Spline1D(InterpolatedUnivariateSpline):

    def __init__(self, x_range, f):
        """

        :param x_range: 1 dimensional array
        :param f: 1-dimensional array
        """

        super(Spline1D, self).__init__(x_range, f, k=3, ext=3)

    def ev_all(self, x):
        return self.ev_energy(x), self.ev_forces(x)

    def ev_forces(self, x):
        fs_scalars = super(Spline1D, self).__call__(x, nu=1)
        return fs_scalars

    def ev_energy(self, x):
        energy_single = super(Spline1D, self).__call__(x, nu=0)
        return energy_single

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        x_range, energies = data['x'], data['f']
        return cls(x_range, energies)

    def save(self, filename):
        np.savez_compressed(filename, x=self._data[0], f=self._data[1])


if __name__ == '__main__':
    from pathlib import Path

    n = 101
    x = np.linspace(0, 10, n)
    f = np.random.rand(n)

    s1 = Spline1D(x, f)

    filename = Path('grid.npz')

    s1.save(filename)

    s2 = Spline1D.load(filename)

    print(s1.ev_all(3))
    print(s2.ev_all(3))
