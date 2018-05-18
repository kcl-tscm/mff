"""

"""
from abc import ABCMeta
import json


class Parameters(object):
    def __init__(self):
        pass


class Model(metaclass=ABCMeta):

    def __init__(self, r_cut):
        self.r_cut = r_cut

    def save(self, jsonfile):
        #
        parameters = {
            'parameters': {
                'cutoff': self.r_cut,
                'elements': [11]

            },
            'configurations:': {
                'target_number': 3000

            },
            'gaussianprocess': {
                'noise_level': 1e-4,

            },
            'remappedpotential': {
                'type': 'single_species_two_body',
                'r_min': 1.5,
                'r_max': 3.2,
                'r_len': 12,
                'filenames': {
                    '1_1': 'data/grid_1_1.npy',
                    '1_1_1': 'data/grid_1_1_1.npy',
                }
            }
        }

        with open(jsonfile, 'w') as file:
            json.dump(parameters, file, sort_keys=False, indent=4)

    @classmethod
    def load(cls, jsonfile):
        with open(jsonfile, 'r') as file:
            parameters = json.load(file)

        return cls(parameters)

    def __str__(self):
        out = '\n'.join([
            'Parameters:',
            '  cutoff: {}'.format(self.r_cut)
        ])

        return out


class SingleElementForces(Model):
    pass


class TwoElementForces(Model):
    pass


default0_json = {
    'remapped_potential': {
        'name': 'single_species_three_body',
        'r_min': 1.5,
        'r_cut': 3.2,
        'r_len': 12,
        'atomic_number': 11,
        "filenames": {
            '1_1': 'data/grid_1_1.npy',
            '1_1_1': 'data/grid_1_1_1.npy',
        }
    }
}

default1_json = {
    'remapped_potential': {
        'name': 'two_species_three_body',
        'r_min': 1.5,
        'r_cut': 3.2,
        'r_len': 12,
        'elements': [11, 12],
        "filenames": {
            '1_1': 'data/grid_1_1.npy',
            '1_2': 'data/grid_1_2.npy',
            '2_2': 'data/grid_2_2.npy',
            '1_1_1': 'data/grid_1_1_1.npy',
            '1_1_2': 'data/grid_1_1_2.npy',
            '1_2_2': 'data/grid_1_2_2.npy',
            '2_2_2': 'data/grid_2_2_2.npy'
        }
    }
}

if __name__ == '__main__':
    model = SingleElementForces(3.5)
    print(model)

    filename = 'test.json'
    model.save(filename)

# Future structure of json file:
# {
#     "nbodies": 2,
#     "elements": [11],
#
#     "configurations:": {
#         "cutoff": 3.5,
#         "sampling method": "linspace",
#         "target number": 3000
#     },
#     "gaussianprocess": {
#         "ntraining": 3,
#         "cutoff radius": 3.5,
#         "cutoff decay": 0.4,
#         "lengthscale": 1.0,
#         "noise level": 0.0001,
#         "mean prior": 0.4,
#         "filename": {
#             "coefs":''
#         }
#     },
#     "remappedpotential": {
#         "r_min": 1.5,
#         "r_max": 3.5,
#         "r_len": 12,
#         "filenames": {
#             "1_1": "data/grid_1_1.npy",
#             "1_1_1": "data/grid_1_1_1.npy"
#         }
#     }
# }
#
