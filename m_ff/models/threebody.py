import json
import numpy as np
import warnings

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation

from .base import ThreeBodyModel, SingleSpeciesModel, TwoSpeciesModel


class ThreeBodySingleSpeciesModel(ThreeBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        kernel = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

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

    def build_grid(self, start, num):
        """Function that builds and predicts energies on a cube of values"""

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
        grid_data[inds] = self.predict_energy(confs).flatten()

        for ind_i in range(num):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_data[ind_i, ind_k, ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_i, ind_k] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_k, ind_i] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_i, ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_j, ind_i] = grid_data[ind_i, ind_j, ind_k]

        grid = interpolation.Spline3D(dists, dists, dists, grid_data)

        self.grid[(self.element, self.element, self.element)] = grid

    def save_grid(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        grid = self.grid[(self.element, self.element, self.element)]
        np.save(filename, grid)
        print('Saved 3-body grid  with name:', filename)

    def load_grid(self, filename):
        warnings.warn('use save and load function', DeprecationWarning)
        grid = np.load(filename)
        self.grid[(self.element, self.element, self.element)] = grid
        print('Loaded 3-body grid from file:', filename)

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


class ThreeBodyTwoSpeciesModel(TwoSpeciesModel):

    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(elements, r_cut)

        kernel = kernels.ThreeBodyTwoSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

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

    def build_grid(self, start, num):
        """Function that builds and predicts energies on a cube of values"""

        dists = np.linspace(start, self.r_cut, num)

        grid_0_0_0 = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[0])
        grid_0_0_1 = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[1])
        grid_0_1_1 = self.build_grid_3b(dists, self.elements[0], self.elements[1], self.elements[1])
        grid_1_1_1 = self.build_grid_3b(dists, self.elements[1], self.elements[1], self.elements[1])

        self.grid[(self.elements[0], self.elements[0], self.elements[0])] = grid_0_0_0
        self.grid[(self.elements[0], self.elements[0], self.elements[1])] = grid_0_0_1
        self.grid[(self.elements[0], self.elements[1], self.elements[1])] = grid_0_1_1
        self.grid[(self.elements[1], self.elements[1], self.elements[1])] = grid_1_1_1

        # return grid_1_1_1, grid_1_1_2, grid_1_2_2, grid_2_2_2

    def build_grid_3b(self, dists, element_k, element_i, element_j):
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
        grid_3b[inds] = self.predict_energy(confs).flatten()

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
