# -*- coding: utf-8 -*-
from abc import ABCMeta

# import json
import numpy as np

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation


class Model(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grid = dict()

    def savejson(self, directory, prefix):
        pass

    def saveall(self, directory, prefix):
        pass

    @classmethod
    def load(cls):
        pass


class TwoBodyModel(Model, metaclass=ABCMeta):
    pass


class ThreeBodyModel(Model, metaclass=ABCMeta):
    pass


class CombinedModel(Model, metaclass=ABCMeta):
    pass


class SingleSpeciesModel(Model, metaclass=ABCMeta):

    def __init__(self, element, r_cut):
        super().__init__()

        self.element = element
        self.r_cut = r_cut


class TwoSpeciesModel(Model, metaclass=ABCMeta):
    def __init__(self, elements, r_cut):
        super().__init__()

        self.elements = elements
        self.r_cut = r_cut


class TwoBodySingleSpeciesModel(TwoBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        kernel = kernels.TwoBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
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
        self.gp.save(filename)

    def load_gp(self, filename):
        self.gp.load(filename)

    def build_grid(self, start, num):
        dists = np.linspace(start, self.r_cut, num)

        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element

        grid_data = self.gp.predict_energy(confs)
        grid = interpolation.Spline1D(dists, grid_data)

        self.grid[(self.element, self.element)] = grid


class ThreeBodySingleSpeciesModel(ThreeBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        kernel = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def fit_energy(self, confs, forces):
        self.gp.fit_energy(confs, forces)

    def fit_force_and_energy(self, confs, forces, energies):
        self.gp.fit_force_and_energy(confs, forces, energies)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def save_gp(self, filename):
        self.gp.save(filename)

    def load_gp(self, filename):
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


class CombinedSingleSpeciesModel(ThreeBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, **kwargs):
        super().__init__(element, r_cut)

        kernel_2b = kernels.TwoBodySingleSpeciesKernel(theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(kernel=kernel_3b, noise=noise, **kwargs)

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

    def build_grid(self, start, num_2b, num_3b):
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

        self.grid[(self.element, self.element)] = grid_2b
        self.grid[(self.element, self.element, self.element)] = grid_3b

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


class TwoBodyTwoSpeciesModel(TwoSpeciesModel):
    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(elements, r_cut)

        kernel = kernels.TwoBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
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
        self.gp.save(filename)

    def load_gp(self, filename):
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

    def build_grid_3b(self, dists, element_i, element_j, element_k):
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


class CombinedTwoSpeciesModel(TwoSpeciesModel):
    pass

# class Parameters(object):
#     def __init__(self):
#         pass
#
#
# def load():
#     pass
#
#
# class Model(metaclass=ABCMeta):
#
#     def __init__(self, r_cut):
#         self.r_cut = r_cut
#
#     def save(self, jsonfile):
#         #
#         parameters = {
#             'parameters': {
#                 'cutoff': self.r_cut,
#                 'elements': [11]
#
#             },
#             'configurations:': {
#                 'target_number': 3000
#
#             },
#             'gaussianprocess': {
#                 'noise_level': 1e-4,
#
#             },
#             'remappedpotential': {
#                 'type': 'single_species_two_body',
#                 'r_min': 1.5,
#                 'r_max': 3.2,
#                 'r_len': 12,
#                 'filenames': {
#                     '1_1': 'data/grid_1_1.npy',
#                     '1_1_1': 'data/grid_1_1_1.npy',
#                 }
#             }
#         }
#
#         with open(jsonfile, 'w') as file:
#             json.dump(parameters, file, sort_keys=False, indent=4)
#
#     @classmethod
#     def load(cls, jsonfile):
#         with open(jsonfile, 'r') as file:
#             parameters = json.load(file)
#
#         return cls(parameters)
#
#     def __str__(self):
#         out = '\n'.join([
#             'Parameters:',
#             '  cutoff: {}'.format(self.r_cut)
#         ])
#
#         return out
#
#
# class SingleElementForces(Model):
#     pass
#
#
# class TwoElementForces(Model):
#     pass
#
#
# default0_json = {
#     'remapped_potential': {
#         'name': 'single_species_three_body',
#         'r_min': 1.5,
#         'r_cut': 3.2,
#         'r_len': 12,
#         'atomic_number': 11,
#         "filenames": {
#             '1_1': 'data/grid_1_1.npy',
#             '1_1_1': 'data/grid_1_1_1.npy',
#         }
#     }
# }
#
# default1_json = {
#     'remapped_potential': {
#         'name': 'two_species_three_body',
#         'r_min': 1.5,
#         'r_cut': 3.2,
#         'r_len': 12,
#         'elements': [11, 12],
#         "filenames": {
#             '1_1': 'data/grid_1_1.npy',
#             '1_2': 'data/grid_1_2.npy',
#             '2_2': 'data/grid_2_2.npy',
#             '1_1_1': 'data/grid_1_1_1.npy',
#             '1_1_2': 'data/grid_1_1_2.npy',
#             '1_2_2': 'data/grid_1_2_2.npy',
#             '2_2_2': 'data/grid_2_2_2.npy'
#         }
#     }
# }
#
# if __name__ == '__main__':
#     model = SingleElementForces(3.5)
#     print(model)
#
#     filename = 'test.json'
#     model.save(filename)
#
# # Future structure of json file:
# # {
# #     "nbodies": 2,
# #     "elements": [11],
# #
# #     "configurations:": {
# #         "cutoff": 3.5,
# #         "sampling method": "linspace",
# #         "target number": 3000
# #     },
# #     "gaussianprocess": {
# #         "ntraining": 3,
# #         "cutoff radius": 3.5,
# #         "cutoff decay": 0.4,
# #         "lengthscale": 1.0,
# #         "noise level": 0.0001,
# #         "mean prior": 0.4,
# #         "filename": {
# #             "coefs":''
# #         }
# #     },
# #     "remappedpotential": {
# #         "r_min": 1.5,
# #         "r_max": 3.5,
# #         "r_len": 12,
# #         "filenames": {
# #             "1_1": "data/grid_1_1.npy",
# #             "1_1_1": "data/grid_1_1_1.npy"
# #         }
# #     }
# # }
# #
