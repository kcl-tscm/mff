import json
import numpy as np
import warnings
from pathlib import Path

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation

from m_ff.models.base import Model


class ThreeBodySingleSpeciesModel(Model):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()
        self.element = element
        self.r_cut = r_cut

        kernel = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

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

    def build_grid(self, start, num, nnodes = 1):
        """Function that builds and predicts energies on a cube of values"""
        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))
        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        # Permutations of elements
        confs[:, :, 3] = self.element  # Central element is always element 1
        confs[:, 0, 4] = self.element  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element  # Element on the xy plane is always element 3

        grid_data = np.zeros((num, num, num))

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)
            import sys
            sys.setrecursionlimit(1000000)
            print('Using %i cores for the mapping' % (nnodes))
            pool = ProcessingPool(nodes=nnodes)
            splitind = np.zeros(nnodes + 1)
            factor = (n + (nnodes - 1)) / nnodes
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
            result = np.array(pool.map(self.gp.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_data[inds] = result
            
        else:
            grid_data[inds] = self.predict_energy(confs).flatten()
            
        for ind_i in range(num):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_data[ind_i, ind_k, ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_i, ind_k] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_k, ind_i] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_i, ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_j, ind_i] = grid_data[ind_i, ind_j, ind_k]

        self.grid = interpolation.Spline3D(dists, dists, dists, grid_data)

    @staticmethod
    def generate_triplets(dists):
        d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Utilizing permutation invariance
        inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y

    def save(self, path):

        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem

        params = {
            'model': self.__class__.__name__,
            'elements': self.element,
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

        gp_filename = "{}_gp_ker_3_ntr_{p[gp][n_train]}".format(prefix, p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(directory / gp_filename)

        if self.grid:
            grid_filename = '{}_grid_num_{p[grid][r_num]}'.format(prefix, p=params)

            params['grid']['filename'] = grid_filename
            self.grid.save(directory / grid_filename)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)


class ThreeBodyTwoSpeciesModel(Model):

    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()
        self.elements = elements
        self.r_cut = r_cut

        kernel = kernels.ThreeBodyTwoSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = {}, None, None

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def fit_energy(self, confs, forces):
        self.gp.fit_energy(confs, forces)

    def fit_force_and_energy(self, confs, forces, energies):
        self.gp.fit_force_and_energy(confs, forces, energies)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def save_gp(self, filename):
        self.gp.save(filename)

    def load_gp(self, filename):
        self.gp.load(filename)

    def build_grid(self, start, num, nnodes = 1):
        """Function that builds and predicts energies on a cube of values"""
        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)

        self.grid[(0, 0, 0)] = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[0], nnodes)
        self.grid[(0, 0, 1)] = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[1], nnodes)
        self.grid[(0, 1, 1)] = self.build_grid_3b(dists, self.elements[0], self.elements[1], self.elements[1], nnodes)
        self.grid[(1, 1, 1)] = self.build_grid_3b(dists, self.elements[1], self.elements[1], self.elements[1], nnodes)

    def build_grid_3b(self, dists, element_k, element_i, element_j, nnodes):
#     def build_grid_3b(self, dists, element_i, element_j, element_k, nnodes):

        # HOTFIX: understand why this weird order is correct
        """Function that builds and predicts energies on a cube of values"""

        num = len(dists)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets_all(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))

        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        confs[:, :, 3] = element_i  # Central element is always element 1
        confs[:, 0, 4] = element_j  # Element on the x axis is always element 2
        confs[:, 1, 4] = element_k  # Element on the xy plane is always element 3

        grid_3b = np.zeros((num, num, num))

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)
            import sys
            sys.setrecursionlimit(1000000)
            print('Using %i cores for the mapping' % (nnodes))
            pool = ProcessingPool(nodes=nnodes)
            splitind = np.zeros(nnodes + 1)
            factor = (n + (nnodes - 1)) / nnodes
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
            result = np.array(pool.map(self.gp.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_3b[inds] = result
            
        else:
            grid_3b[inds] = self.gp.predict_energy(confs).flatten()

        # for ind_i in range(num):
        #     for ind_j in range(ind_i + 1):
        #         for ind_k in range(ind_j + 1):
        #             grid_3b[ind_i, ind_k, ind_j] = grid_3b[ind_i, ind_j, ind_k]
        #             grid_3b[ind_j, ind_i, ind_k] = grid_3b[ind_i, ind_j, ind_k]
        #             grid_3b[ind_j, ind_k, ind_i] = grid_3b[ind_i, ind_j, ind_k]
        #             grid_3b[ind_k, ind_i, ind_j] = grid_3b[ind_i, ind_j, ind_k]
        #             grid_3b[ind_k, ind_j, ind_i] = grid_3b[ind_i, ind_j, ind_k]

        return interpolation.Spline3D(dists, dists, dists, grid_3b)

    @staticmethod
    def generate_triplets_with_permutation_invariance(dists):
        d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Utilizing permutation invariance
        inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y

    @staticmethod
    def generate_triplets_all(dists):
        d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y

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

        gp_filename = "{}_gp_ker_3_ntr_{p[gp][n_train]}".format(prefix, p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(directory / gp_filename)

        params['grid']['filename'] = {}
        for k, grid in self.grid.items():
            key = '_'.join(str(element) for element in k)
            grid_filename = '{}_grid_{}_num_{p[grid][r_num]}'.format(prefix, key, p=params)

            params['grid']['filename'][key] = grid_filename
            self.grid[k].save(directory / grid_filename)

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
        model.gp.load(directory / gp_filename)

        if params['grid']:            

            for key, grid_filename in params['grid']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid[k] = interpolation.Spline3D.load(directory / grid_filename)
            
            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

        return model

if __name__ == '__main__':
    def test_three_body_single_species_model():
        confs = np.array([
            np.hstack([np.random.randn(4, 3), 26 * np.ones((4, 2))]),
            np.hstack([np.random.randn(5, 3), 26 * np.ones((5, 2))])
        ])

        forces = np.random.randn(2, 3)

        element, r_cut, sigma, theta, noise = 26, 2., 3., 4., 5.

        filename = Path() / 'test_model'

        m = ThreeBodySingleSpeciesModel(element, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)


    def test_three_body_two_species_model():
        elements = [2, 4]
        confs = np.array([
            np.hstack([np.random.randn(4, 3), np.random.choice(elements, size=(4, 2))]),
            np.hstack([np.random.randn(5, 3), np.random.choice(elements, size=(5, 2))])
        ])

        forces = np.random.randn(2, 3)
        r_cut, sigma, theta, noise = 2., 3., 4., 5.

        filename = Path() / 'test_model'

        m = ThreeBodyTwoSpeciesModel(elements, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)


    test_three_body_single_species_model()
    # test_three_body_two_species_model()
