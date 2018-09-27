import json
import numpy as np
import warnings
from pathlib import Path

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation

from m_ff.models.base import Model
import warnings


class TwoBodySingleSpeciesModel(Model):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()

        self.element = element
        self.r_cut = r_cut

        kernel = kernels.TwoBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

    def fit(self, confs, forces, nnodes = 1):
        self.gp.fit(confs, forces, nnodes)

    def fit_energy(self, confs, energies, nnodes = 1):
        self.gp.fit_energy(confs, energies, nnodes)

    def fit_force_and_energy(self, confs, forces, energies, nnodes = 1):
        self.gp.fit_force_and_energy(confs, forces, energies, nnodes)

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

    def build_grid(self, start, num):
        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)

        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element

        grid_data = self.gp.predict_energy(confs)
        self.grid = interpolation.Spline1D(dists, grid_data)

    def save(self, path):

        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem

        params = {
            'model': self.__class__.__name__,
            'element': self.element,
            'r_cut': self.r_cut,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {
                'r_min': self.grid_start,
                'r_num': self.grid_num
            } if self.grid else {}
        }

        gp_filename = "{}_gp_ker_2_ntr_{p[gp][n_train]}.npy".format(prefix, p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(directory / gp_filename)

        if self.grid:
            grid_filename = '{}_grid_num_{p[grid][r_num]}.npz'.format(prefix, p=params)

            params['grid']['filename'] = grid_filename
            self.grid.save(directory / grid_filename)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)

    @classmethod
    def from_json(cls, path):
        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem

        with open(path) as fp:
            params = json.load(fp)

        model = cls(params['element'],
                    params['r_cut'],
                    params['gp']['sigma'],
                    params['gp']['theta'],
                    params['gp']['noise'])

        gp_filename = params['gp']['filename']
        model.gp.load(directory / gp_filename)

        if params['grid']:
            grid_filename = params['grid']['filename']
            model.grid = interpolation.Spline1D.load(directory / grid_filename)

            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

        return model


class TwoBodyTwoSpeciesModel(Model):
    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()

        self.elements = elements
        self.r_cut = r_cut

        kernel = kernels.TwoBodyTwoSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = {}, None, None

    def fit(self, confs, forces, nnodes = 1):
        self.gp.fit(confs, forces, nnodes)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def fit_energy(self, confs, energy, nnodes = 1):
        self.gp.fit_energy(confs, energy, nnodes)

    def fit_force_and_energy(self, confs, forces, energy, nnodes = 1):
        self.gp.fit_force_and_energy(confs, forces, energy, nnodes)

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

        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)

        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[0], self.elements[0]
        grid_0_0_data = self.gp.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[0], self.elements[1]
        grid_0_1_data = self.gp.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[1], self.elements[1]
        grid_1_1_data = self.gp.predict_energy(confs)

        self.grid[(0, 0)] = interpolation.Spline1D(dists, grid_0_0_data)
        self.grid[(0, 1)] = interpolation.Spline1D(dists, grid_0_1_data)
        self.grid[(1, 1)] = interpolation.Spline1D(dists, grid_1_1_data)

    def save(self, path):

        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem

        params = {
            'model': self.__class__.__name__,
            'elements': self.elements,
            'r_cut': self.r_cut,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {
                'r_min': self.grid_start,
                'r_num': self.grid_num
            } if self.grid else {}
        }

        gp_filename = "{}_gp_ker_2_ntr_{p[gp][n_train]}.npy".format(prefix, p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(directory / gp_filename)

        params['grid']['filename'] = {}
        for k, grid in self.grid.items():
            key = '_'.join(str(element) for element in k)
            grid_filename = '{}_grid_{}_num_{p[grid][r_num]}.npz'.format(prefix, key, p=params)

            params['grid']['filename'][key] = grid_filename
            grid[k].save(directory / grid_filename)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)

    @classmethod
    def from_json(cls, path):
        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem

        with open(path) as fp:
            params = json.load(fp)

        model = cls(params['elements'],
                    params['r_cut'],
                    params['gp']['sigma'],
                    params['gp']['theta'],
                    params['gp']['noise'])

        gp_filename = params['gp']['filename']
        try:
            model.gp.load(directory / gp_filename)
        except:
            warnings.warn("The 2-body GP file is missing")
            pass

        if params['grid']:
            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

            for key, grid_filename in params['grid']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))

                model.grid[k] = interpolation.Spline1D.load(directory / grid_filename)

        return model


if __name__ == '__main__':
    def test_two_body_single_species_model():
        confs = np.array([
            np.hstack([np.random.randn(4, 3), 26 * np.ones((4, 2))]),
            np.hstack([np.random.randn(5, 3), 26 * np.ones((5, 2))])
        ])

        forces = np.random.randn(2, 3)

        element, r_cut, sigma, theta, noise = 26, 2., 3., 4., 5.

        filename = Path('test_model.json')

        m = TwoBodySingleSpeciesModel(element, r_cut, sigma, theta, noise)

        print(m)
        print(m.parameters)

        m.fit(confs, forces)

        print(m)
        print(m.parameters)

        m.build_grid(1., 10)

        print(m)
        print(m.parameters)

        m.save(filename)

        m2 = TwoBodySingleSpeciesModel.from_json(filename)


    def test_two_body_two_species_model():
        elements = [2, 4]
        confs = np.array([
            np.hstack([np.random.randn(4, 3), np.random.choice(elements, size=(4, 2))]),
            np.hstack([np.random.randn(5, 3), np.random.choice(elements, size=(5, 2))])
        ])

        forces = np.random.randn(2, 3)
        r_cut, sigma, theta, noise = 2., 3., 4., 5.

        filename = Path('test_model.json')

        m = TwoBodyTwoSpeciesModel(elements, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)

        m2 = TwoBodyTwoSpeciesModel.from_json(filename)



    # test_two_body_single_species_model()

    test_two_body_two_species_model()
