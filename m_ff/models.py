"""

"""
from abc import ABCMeta

# import json
import numpy as np

from m_ff import gp


class Model(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.element = elements
        self.r_cut = r_cut


class TwoBodySingleSpeciesModel(TwoBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        self.gp = gp.TwoBodySingleSpeciesGP(theta=[sigma, theta, r_cut], noise=noise, **kwargs)

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def build_grid(self, start, num):
        dists = np.linspace(start, self.r_cut, num)
        grid = self.gp.build_grid(dists, self.element)

        return grid


class ThreeBodySingleSpeciesModel(ThreeBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        self.gp = gp.ThreeBodySingleSpeciesGP(theta=[sigma, theta, r_cut], noise=noise, **kwargs)

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def build_grid(self, start, num):
        """Function that builds and predicts energies on a cube of values"""
        dists = np.linspace(start, self.r_cut, num)

        grid = self.gp.build_grid(dists, self.element)

        return grid


class CombinedSingleSpeciesModel(ThreeBodyModel, SingleSpeciesModel):

    def __init__(self, element, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, **kwargs):
        super().__init__(element, r_cut)

        self.gp_2b = gp.TwoBodySingleSpeciesGP(theta=[sigma_2b, theta_2b, r_cut], noise=noise, **kwargs)
        self.gp_3b = gp.ThreeBodySingleSpeciesGP(theta=[sigma_3b, theta_3b, r_cut], noise=noise, **kwargs)

    def fit(self, confs, forces):
        self.gp_2b.fit(confs, forces)

        ntr = len(confs)
        two_body_forces = np.zeros((ntr, 3))
        for i in np.arange(ntr):
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit(confs, forces - two_body_forces)

    def predict(self, confs, return_std=False):
        return self.gp_2b.predict(confs, return_std) + \
               self.gp_3b.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp_2b.predict_energy(confs, return_std) + \
               self.gp_3b.predict_energy(confs, return_std)

    def build_grid(self, start, num_2b, num_3b):
        """Function that builds and predicts energies on a cube of values"""
        dists_2b = np.linspace(start, self.r_cut, num_2b)
        dists_3b = np.linspace(start, self.r_cut, num_3b)

        grid_2b = self.gp_2b.build_grid(dists_2b, self.element)
        grid_3b = self.gp_3b.build_grid(dists_3b, self.element)

        return grid_2b, grid_3b


class TwoBodyTwoSpeciesModel(TwoSpeciesModel):
    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__(element, r_cut)

        self.gp = gp.TwoBodyTwoSpeciesGP(theta=[sigma, theta, r_cut], noise=noise, **kwargs)

    def fit(self, confs, forces):
        self.gp.fit(confs, forces)

    def predict(self, confs, return_std=False):
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        return self.gp.predict_energy(confs, return_std)

    def build_grid(self, start, num):
        """Function that builds and predicts energies on a cube of values"""

        dists = np.linspace(start, self.r_cut, num)
        grid = self.gp.build_grid(dists, self.element)

        return grid


class ThreeBodyTwoSpeciesModel(SingleSpeciesModel):
    pass


class CombinedTwoSpeciesModel(SingleSpeciesModel):
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
