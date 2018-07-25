import json
import numpy as np
import warnings

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation

from .base import TwoBodyModel, SingleSpeciesModel, TwoSpeciesModel


class TwoBodySingleSpeciesModel(TwoBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        kernel = kernels.TwoBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid_start, self.grid_num = None, None

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def fit_energy(self, confs, energies):
        self.gp.fit_energy(confs, energies)

    def fit_force_and_energy(self, confs, forces, energies):
        self.gp.fit_force_and_energy(confs, forces, energies)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def save_gp(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.save(filename)

    def load_gp(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.load(filename)

    def build_grid(self, r_min, num):
        self.grid_start = r_min
        self.grid_num = num

        dists = np.linspace(r_min, self.r_cut, num)

        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element

        grid_data = self.gp.predict_energy(confs)
        grid = interpolation.Spline1D(dists, grid_data)

        self.grid[(self.element, self.element)] = grid

    def save_grid(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        grid = self.grid[(self.element, self.element)]
        np.save(filename, grid)
        print('Saved 2-body grid  with name:', filename)

    def load_grid(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        grid = np.load(filename)
        self.grid[(self.element, self.element)] = grid
        print('Loaded 2-body grid from file:', filename)

    @property
    def parameters(self):
        params = super().parameters
        params['model'] = self.__class__.__name__
        params['gp_noise'] = self.gp.noise
        params['gp_kernel'] = self.gp.kernel.kernel_name
        params['gp_sigma'], params['gp_theta'], _ = self.gp.kernel.theta

        if self.grid:
            params['r_min'] = self.grid_start
            params['r_num'] = self.grid_num

        return params


class TwoBodyTwoSpeciesModel(TwoSpeciesModel):
    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(elements, r_cut)

        kernel = kernels.TwoBodyTwoSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def fit_energy(self, confs, energy):
        self.gp.fit_energy(confs, energy)

    def fit_force_and_energy(self, confs, forces, energy):
        self.gp.fit_force_and_energy(confs, forces, energy)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def save_gp(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.save(filename)

    def load_gp(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.load(filename)

    def build_grid(self, start, num):
        """Function that builds and predicts energies on a cube of values"""

        dists = np.linspace(start, self.r_cut, num)

        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists

        confs[:, 0, 3], confs[:, 0, 4] = self.elements
        grid_0_0_data = self.gp.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements
        grid_0_1_data = self.gp.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements
        grid_1_1_data = self.gp.predict_energy(confs)

        grid_0_0 = interpolation.Spline1D(dists, grid_0_0_data)
        grid_0_1 = interpolation.Spline1D(dists, grid_0_1_data)
        grid_1_1 = interpolation.Spline1D(dists, grid_1_1_data)

        self.grid[(self.elements[0], self.elements[0])] = grid_0_0
        self.grid[(self.elements[0], self.elements[1])] = grid_0_1
        self.grid[(self.elements[1], self.elements[1])] = grid_1_1

    def save(self, directory, prefix):
        pass

    def load(self, directory, prefix):
        pass
