import json
import numpy as np
import warnings
from pathlib import Path

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation

from .base import Model

        
class CombinedSingleSpeciesModel(Model):

    def __init__(self, element, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, **kwargs):
        super().__init__()
        self.element = element
        self.r_cut = r_cut

        kernel_2b = kernels.TwoBodySingleSpeciesKernel(theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(kernel=kernel_3b, noise=noise, **kwargs)

        self.grid_2b, self.grid_3b, self.grid_start, self.grid_num = None, None, None, None

    def fit(self, confs, forces):
        self.gp_2b.fit(confs, forces)

        ntr = len(confs)
        two_body_forces = np.zeros((ntr, 3))
        for i in np.arange(ntr):
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit(confs, forces - two_body_forces)

    def fit_energy(self, confs, energies):
        self.gp_2b.fit_energy(confs, energies)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit_energy(confs, energies - two_body_energies)

    def fit_force_and_energy(self, confs, forces, energies):
        self.gp_2b.fit_force_and_energy(confs, forces, energies)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        two_body_forces = np.zeros((ntr, 3))

        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))
        self.gp_3b.fit_force_and_energy(confs, forces - two_body_forces, energies - two_body_energies)

    def predict(self, confs, return_std=False):
        return self.gp_2b.predict(confs, return_std) + \
               self.gp_3b.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp_2b.predict_energy(confs, return_std) + \
               self.gp_3b.predict_energy(confs, return_std)

    def build_grid(self, start, num_2b, num_3b, nnodes = 1):
        """Function that builds and predicts energies on a cube of values"""
        dists_2b = np.linspace(start, self.r_cut, num_2b)

        confs = np.zeros((num_2b, 1, 5))
        confs[:, 0, 0] = dists_2b
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element

        grid_data = self.gp_2b.predict_energy(confs)
        grid_2b = interpolation.Spline1D(dists_2b, grid_data)

        # Mapping 3 body part
        dists_3b = np.linspace(start, self.r_cut, num_3b)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets(dists_3b)

        confs = np.zeros((len(r_ij_x), 2, 5))
        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        # Permutations of elements
        confs[:, :, 3] = self.element  # Central element is always element 1
        confs[:, 0, 4] = self.element  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element  # Element on the xy plane is always element 3

        grid_3b = np.zeros((num_3b, num_3b, num_3b))
        
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
            result = np.array(pool.map(self.gp_3b.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_3b[inds] = result
            
        else:
            grid_3b[inds] = self.gp_3b.predict_energy(confs).flatten()
            
        for ind_i in range(num_3b):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_3b[ind_i, ind_k, ind_j] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_j, ind_i, ind_k] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_j, ind_k, ind_i] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_k, ind_i, ind_j] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_k, ind_j, ind_i] = grid_3b[ind_i, ind_j, ind_k]

        grid_3b = interpolation.Spline3D(dists_3b, dists_3b, dists_3b, grid_3b)

        self.grid_2b = grid_2b
        self.grid_3b = grid_3b
        self.grid_num_2b = num_2b
        self.grid_num_3b = num_3b
        self.grid_start = start
        
    def save_combined(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem
        
        ### SAVE THE 2B MODEL ###
        params = {
            'model': self.__class__.__name__,
            'element': self.element,
            'r_cut': self.r_cut,
            'gp_2b': {
                'kernel': self.gp_2b.kernel.kernel_name,
                'n_train': self.gp_2b.n_train,
                'sigma': self.gp_2b.kernel.theta[0],
                'theta': self.gp_2b.kernel.theta[1],
                'noise': self.gp_2b.noise
            },
            'gp_3b': {
                'kernel': self.gp_3b.kernel.kernel_name,
                'n_train': self.gp_3b.n_train,
                'sigma': self.gp_3b.kernel.theta[0],
                'theta': self.gp_3b.kernel.theta[1],
                'noise': self.gp_3b.noise
            },
            'grid_2b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_2b
            } if self.grid_2b else {}
            ,
            'grid_3b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_3b
            } if self.grid_3b else {}
        }

        gp_filename_2b = "{}_gp_ker_2_ntr_{p[gp_2b][n_train]}.npy".format(prefix, p=params)

        params['gp_2b']['filename'] = gp_filename_2b
        self.gp_2b.save(directory / gp_filename_2b)

        if self.grid_2b:
            grid_filename_2b = '{}_grid_2b_num_{p[grid_2b][r_num]}.npz'.format(prefix, p=params)
            print("Saved 2-body grid under name %s" %(grid_filename_2b))
            params['grid_2b']['filename'] = grid_filename_2b
            self.grid_2b.save(directory / grid_filename_2b)
        
        ### SAVE THE 3B MODEL ###
        gp_filename_3b = "{}_gp_ker_3_ntr_{p[gp_3b][n_train]}.npy".format(prefix, p=params)

        params['gp_3b']['filename'] = gp_filename_3b
        self.gp_3b.save(directory / gp_filename_3b)

        if self.grid_3b:
            grid_filename_3b = '{}_grid_3b_num_{p[grid_3b][r_num]}.npz'.format(prefix, p=params)
            print("Saved 3-body grid under name %s" %(grid_filename_3b))  
            params['grid_3b']['filename'] = grid_filename_3b
            self.grid_3b.save(directory / grid_filename_3b)

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
                    params['gp_2b']['sigma'],
                    params['gp_3b']['sigma'],
                    params['gp_2b']['theta'],
                    params['gp_3b']['theta'],
                    params['gp_2b']['noise'])

        gp_filename_2b = params['gp_2b']['filename']
        gp_filename_3b = params['gp_3b']['filename']
        model.gp_2b.load(directory / gp_filename_2b)
        model.gp_3b.load(directory / gp_filename_3b)

        if params['grid_2b']:
            grid_filename_2b = params['grid_2b']['filename']
            model.grid_2b = interpolation.Spline1D.load(directory / grid_filename_2b)

            grid_filename_3b = params['grid_3b']['filename']
            model.grid_3b = interpolation.Spline3D.load(directory / grid_filename_3b)
            
            model.grid_start = params['grid_2b']['r_min']
            model.grid_num_2b = params['grid_2b']['r_num']
            model.grid_num_3b = params['grid_3b']['r_num']

        return model
    
    def save_gp(self, filename_2b, filename_3b):
        warnings.warn('use save and load function', DeprecationWarning)
        self.gp_2b.save(filename_2b)
        self.gp_3b.save(filename_3b)

    def load_gp(self, filename_2b, filename_3b):
        warnings.warn('use save and load function', DeprecationWarning)
        self.gp_2b.load(filename_2b)
        self.gp_3b.load(filename_3b)

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


class CombinedTwoSpeciesModel(Model):
    #### TO BE CHECKED ####
    def __init__(self, elements, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, **kwargs):
        super().__init__()
        self.elements = elements
        self.r_cut = r_cut

        kernel_2b = kernels.TwoBodyTwoSpeciesKernel(theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodyTwoSpeciesKernel(theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(kernel=kernel_3b, noise=noise, **kwargs)

        self.grid_2b, self.grid_3b, self.grid_start, self.grid_num_2b, self.grid_num_3b = {}, {}, None, None, None

    def fit(self, confs, forces):
        self.gp_2b.fit(confs, forces)

        ntr = len(confs)
        two_body_forces = np.zeros((ntr, 3))
        for i in np.arange(ntr):
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit(confs, forces - two_body_forces)

    def fit_energy(self, confs, energies):
        self.gp_2b.fit_energy(confs, energies)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit_energy(confs, energies - two_body_energies)

    def fit_force_and_energy(self, confs, forces, energies):
        self.gp_2b.fit_force_and_energy(confs, forces, energies)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        two_body_forces = np.zeros((ntr, 3))

        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))
        self.gp_3b.fit_force_and_energy(confs, forces - two_body_forces, energies - two_body_energies)

    def predict(self, confs, return_std=False):
        return self.gp_2b.predict(confs, return_std) + \
               self.gp_3b.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp_2b.predict_energy(confs, return_std) + \
               self.gp_3b.predict_energy(confs, return_std)

    def build_grid(self, start, num_2b, num_3b, nnodes = 1):
        """Function that builds and predicts energies on a cube of values"""
        self.grid_start = start
        self.grid_num_2b = num_2b
        self.grid_num_3b = num_2b

        dists = np.linspace(start, self.r_cut, num_2b)

        confs = np.zeros((num_2b, 1, 5))
        confs[:, 0, 0] = dists

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[0], self.elements[0]
        grid_0_0_data = self.gp_2b.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[0], self.elements[1]
        grid_0_1_data = self.gp_2b.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[1], self.elements[1]
        grid_1_1_data = self.gp_2b.predict_energy(confs)

        self.grid_2b[(0, 0)] = interpolation.Spline1D(dists, grid_0_0_data)
        self.grid_2b[(0, 1)] = interpolation.Spline1D(dists, grid_0_1_data)
        self.grid_2b[(1, 1)] = interpolation.Spline1D(dists, grid_1_1_data)

        dists = np.linspace(start, self.r_cut, num_3b)


        grid_0_0_0 = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[0], nnodes)
        grid_0_0_1 = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[1], nnodes)
        grid_0_1_1 = self.build_grid_3b(dists, self.elements[0], self.elements[1], self.elements[1], nnodes)
        grid_1_1_1 = self.build_grid_3b(dists, self.elements[1], self.elements[1], self.elements[1], nnodes)

        self.grid_3b[(0, 0, 0)] = grid_0_0_0
        self.grid_3b[(0, 0, 1)] = grid_0_0_1
        self.grid_3b[(0, 1, 1)] = grid_0_1_1
        self.grid_3b[(1, 1, 1)] = grid_1_1_1

    def build_grid_3b(self, dists, element_k, element_i, element_j, nnodes):
        # HOTFIX: understand why this weird order is correct
        """Function that builds and predicts energies on a cube of values"""

        num = len(dists)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets_all(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))

        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        # Permutations of elements

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
            result = np.array(pool.map(self.gp_3b.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_3b[inds] = result
            
        else:
            grid_3b[inds] = self.gp_3b.predict_energy(confs).flatten()

        return interpolation.Spline3D(dists, dists, dists, grid_3b)
    
    def save_combined(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem
        
        ### SAVE THE 2B MODEL ###
        params = {
            'model': self.__class__.__name__,
            'elements': self.elements,
            'r_cut': self.r_cut,
            'gp_2b': {
                'kernel': self.gp_2b.kernel.kernel_name,
                'n_train': self.gp_2b.n_train,
                'sigma': self.gp_2b.kernel.theta[0],
                'theta': self.gp_2b.kernel.theta[1],
                'noise': self.gp_2b.noise
            },
            'gp_3b': {
                'kernel': self.gp_3b.kernel.kernel_name,
                'n_train': self.gp_3b.n_train,
                'sigma': self.gp_3b.kernel.theta[0],
                'theta': self.gp_3b.kernel.theta[1],
                'noise': self.gp_3b.noise
            },
            'grid_2b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_2b
            } if self.grid_2b else {}
            ,
            'grid_3b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_3b
            } if self.grid_3b else {}
        }

        gp_filename_2b = "{}_gp_ker_2_ntr_{p[gp_2b][n_train]}.npy".format(prefix, p=params)

        params['gp_2b']['filename'] = gp_filename_2b
        self.gp_2b.save(directory / gp_filename_2b)
        
        params['grid_2b']['filename'] = {}
        for k, grid in self.grid_2b.items():
            key = '_'.join(str(element) for element in k)
            grid_filename_2b = '{}_grid_{}_num_{p[grid_2b][r_num]}.npz'.format(prefix, key, p=params)
            print("Saved 2-body grid under name %s" %(grid_filename_2b))
            params['grid_2b']['filename'][key] = grid_filename_2b
            self.grid_2b[k].save(directory / grid_filename_2b)
        
        ### SAVE THE 3B MODEL ###
        gp_filename_3b = "{}_gp_ker_3_ntr_{p[gp_3b][n_train]}.npy".format(prefix, p=params)

        params['gp_3b']['filename'] = gp_filename_3b
        self.gp_3b.save(directory / gp_filename_3b)

        params['grid_3b']['filename'] = {}
        for k, grid in self.grid_3b.items():
            key = '_'.join(str(element) for element in k)
            grid_filename_3b = '{}_grid_{}_num_{p[grid_3b][r_num]}.npz'.format(prefix, key, p=params)

            params['grid_3b']['filename'][key] = grid_filename_3b
            self.grid_3b[k].save(directory / grid_filename_3b)

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
                    params['gp_2b']['sigma'],
                    params['gp_3b']['sigma'],
                    params['gp_2b']['theta'],
                    params['gp_3b']['theta'],
                    params['gp_2b']['noise'])

        gp_filename_2b = params['gp_2b']['filename']
        gp_filename_3b = params['gp_3b']['filename']
        model.gp_2b.load(directory / gp_filename_2b)
        model.gp_3b.load(directory / gp_filename_3b)

        if params['grid_2b']:            
            for key, grid_filename_2b in params['grid_2b']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid_2b[k] = interpolation.Spline1D.load(directory / grid_filename_2b)

            for key, grid_filename_3b in params['grid_3b']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid_3b[k] = interpolation.Spline3D.load(directory / grid_filename_3b)
            
            model.grid_start = params['grid_2b']['r_min']
            model.grid_num_2b = params['grid_2b']['r_num']
            model.grid_num_3b = params['grid_3b']['r_num']

        return model
    
    def save_gp(self, filename_2b, filename_3b):
        self.gp_2b.save(filename_2b)
        self.gp_3b.save(filename_3b)

    def load_gp(self, filename_2b, filename_3b):
        self.gp_2b.load(filename_2b)
        self.gp_3b.load(filename_3b)

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
