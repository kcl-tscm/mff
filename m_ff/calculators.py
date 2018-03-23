"""Remapped potential
==================

Introduction
------------

...

Theory
------

...

Running the Calculator
----------------------

...

Example
-------

...
"""
from abc import ABCMeta
import json
from pathlib import Path
from itertools import combinations, islice

import numpy as np
import logging

from ase.calculators.calculator import Calculator, all_changes
# from ase.neighborlist import NeighborList
from asap3 import FullNeighborList

from m_ff.remapping import Spline1D, Spline3D

logger = logging.getLogger(__name__)


# from pathos.multiprocessing import ProcessingPool
# from ase.data import chemical_symbols, atomic_numbers
# from ase.units import Bohr
# from ase.neighborlist import NeighborList

class SingleSpecies(Exception):
    pass


# If single element, build only a 2- and  3-body grid
# result = [rs, element1, element2, grid_1_1, grid_1_1_1]

# If two elements, build three 2- and four 3-body grids
# result = [rs, element1, element2, grid_1_1, grid_1_2, grid_2_2, grid_1_1_1, grid_1_1_2, grid_1_2_2, grid_2_2_2]


class RemappedPotential(Calculator, metaclass=ABCMeta):

    def __init__(self, r_cut, restart=None, ignore_bad_restart_file=False, label=None, atoms=None, **kwargs):
        super().__init__(restart, ignore_bad_restart_file, label, atoms, **kwargs)

        self.r_cut = r_cut
        self.nl = None

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute.
        """

        super().calculate(atoms, properties, system_changes)

        if 'numbers' in system_changes:
            logger.info('numbers is in system_changes')
            self.initialize(self.atoms)

        self.nl.check_and_update(self.atoms)

    def initialize(self, atoms):
        logger.info('initialize')
        self.nl = FullNeighborList(self.r_cut, atoms=atoms, driftfactor=0.)

    def set(self, **kwargs):
        changed_parameters = super().set(**kwargs)
        if changed_parameters:
            self.reset()


class TwoBodySingleSpecies(RemappedPotential):
    """A remapped 2-body calculator for ase
    """

    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    def __init__(self, r_cut, grid_1_1, restart=None, ignore_bad_restart_file=False, label=None, atoms=None, **kwargs):
        super().__init__(r_cut, restart, ignore_bad_restart_file, label, atoms, **kwargs)

        self.grid_1_1 = grid_1_1

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute.
        """

        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        # todo: proper numpy itereator with enumaration
        for i in range(len(self.atoms)):
            inds, pos, dists = self.nl.get_neighbors(i)

            energy_local = self.grid_1_1(dists, nu=0)
            fs_scalars = - self.grid_1_1(dists, nu=1)

            potential_energies[i] = np.sum(energy_local, axis=0)
            forces[i] = np.sum(fs_scalars.reshape(-1, 1) * pos, axis=0)

        self.results = {'energy': np.sum(potential_energies),
                        'forces': forces}

    @classmethod
    def from_json(cls, filename):
        with open(filename) as file:
            p = json.load(file)

        rs = np.linspace(p['remappedpotential']['r_start'], p['parameters']['cutoff'], p['remappedpotential']['r_num'])

        grid_1_1_filename = p['remappedpotential']['filenames']['grid_1_1']

        # directory =
        rs, element1, element2, grid_1_1, grid_1_1_1 = np.load(directory + grid_1_1_filename)
        # grid_1_1_data = np.load(grid_1_1_filename)

        # grid_1_1 = Spline1D(rs, grid_1_1_data)
        # return cls(r_cut)

    @classmethod
    def from_numpy(cls, filename):
        r_cut = 3.2
        return cls(r_cut)


class ThreeBodySingleSpecies(RemappedPotential):
    """A remapped 3-body calculator for ase
    """

    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    def __init__(self, r_cut, restart=None, ignore_bad_restart_file=False, label=None, atoms=None, **kwargs):
        super().__init__(r_cut, restart, ignore_bad_restart_file, label, atoms, **kwargs)

        self.grid_1_1_1 = Spline3D(np.linspace(0, 10, 100), np.linspace(0, 10, 100), np.linspace(0, 10, 100),
                                   np.linspace(0, 10, 100 ** 3).reshape(100, 100, 100))

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute.
        """

        super().calculate(atoms, properties, system_changes)

        indices, distances, positions = self.find_triplets()

        mapped = self.grid_1_1_1._ev_all(*np.hsplit(distances, 3))

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))
        for (i, j, k), (energy, dE_ij, dE_jk, dE_ki) in zip(indices, mapped):
            forces[i] += positions[(i, j)] * dE_ij + positions[(i, k)] * dE_ki
            forces[j] += positions[(j, k)] * dE_jk + positions[(j, i)] * dE_ij
            forces[k] += positions[(k, i)] * dE_ki + positions[(k, j)] * dE_jk

            potential_energies[[i, j, k]] += energy

        self.results = {'energy': np.sum(potential_energies),
                        'forces': forces}

    def find_triplets(self):
        atoms, nl = self.atoms, self.nl
        # atomic_numbers = self.atoms.get_array('numbers', copy=False)

        indices, distances, positions = [], [], dict()

        for i in range(len(atoms)):

            inds, pos, dists = nl.get_neighbors(i)

            # Limitation
            assert len(inds) is len(np.unique(inds)), "There are repetitive indices!\n{}".format(inds)

            # ignoring already visited atoms
            inds, pos, dists = inds[inds > i], pos[inds > i, :], dists[inds > i]

            for local_ind, (j, pos_ij, dist_ij) in enumerate(zip(inds, pos, dists)):

                # Caching local displacement vectors
                positions[(i, j)], positions[(j, i)] = pos_ij, -pos_ij

                for k, dist_ik in islice(zip(inds, dists), local_ind + 1, None):

                    try:
                        jk_ind = list(nl[j]).index(k)
                    except ValueError:
                        continue  # no valid triplet

                    _, _, dists_j = nl.get_neighbors(j)

                    indices.append([i, j, k])
                    distances.append([dist_ij, dist_ik, dists_j[jk_ind]])

        return np.array(indices), np.sqrt(np.array(distances)), positions

    def find_triplets2(self):
        atoms, nl = self.atoms, self.nl
        indices, distances, positions = [], [], dict()

        # caching
        arr = [nl.get_neighbors(i) for i in range(len(atoms))]

        for i, (inds, pos, dists) in enumerate(arr):
            # assert len(inds) is len(np.unique(inds)), "There are repetitive indices!\n{}".format(inds)

            # ingnore already visited nodes
            inds, pos, dists = inds[inds > i], pos[inds > i, :], dists[inds > i]

            for (j_ind, j), (k_ind, k) in combinations(enumerate(inds[inds > i]), 2):

                jk_ind, = np.where(arr[j][0] == k)

                if not jk_ind.size:
                    continue  # no valid triplet

                indices.append([i, j, k])

                # Caching local position vectors
                positions[(i, j)], positions[(j, i)] = pos[j_ind], -pos[j_ind]
                positions[(i, k)], positions[(k, i)] = pos[k_ind], -pos[k_ind]
                positions[(j, k)], positions[(k, j)] = arr[j][1][jk_ind[0], :], -arr[j][1][jk_ind[0], :]

                distances.append([dists[j_ind], dists[k_ind], arr[j][2][jk_ind[0]]])

        return np.array(indices), np.sqrt(np.array(distances)), positions

    @classmethod
    def from_numpy(cls, filename):
        r_cut = 3.2
        return cls(r_cut)


class TwoBodyTwoSpecies(Calculator):
    pass


class ThreeBodyTwoSpecies(Calculator):
    pass


if __name__ == '__main__':
    # from ase import Atoms
    from ase.io import read

    logging.basicConfig(level=logging.INFO)

    directory = '../test/data/C_a/'
    filename = Path(directory + 'movie.xyz')

    traj = read(str(filename), index=slice(None))
    calc = TwoBodySingleSpecies(r_cut=3.7, grid_1_1=None)

    TwoBodySingleSpecies.from_json(directory + 'test.json')
    # int(1 + (cutoff - grid_start) / grid_spacing)
    rs, element1, element2, grid_1_1, grid_1_1_1 = np.load(directory + 'MFF_2b_ntr_10_sig_1.00_cut_3.70.npy')

    print(rs.shape, element1, element2, grid_1_1.shape)

    # MFF_2b_ntr_10_sig_1.00_cut_3.70
    # rs, element1, element2, grid_1_1, grid_1_1_1
    # test get_forces
    # print('forces for a = {0}'.format(a0))
    # print(calc.get_forces(bulk))
    # # single points for various lattice constants
    # bulk.set_calculator(calc)
    # for n in range(-5, 5, 1):
    #     a = a0 * (1 + n / 100.0)
    #     bulk.set_cell([a] * 3)
    #     print('a : {0} , total energy : {1}'.format(
    #         a, bulk.get_potential_energy()))

# class MLCalculator_local:
#
#     def __init__(self, ff, ef, nl):
#         self.nl = nl
#         self.ff = ff
#         self.ef = ef
#
#     def get_potential_energy(self, atoms):
#         n = len(atoms)
#         nl = self.nl
#         nl.update(atoms)
#         cell = atoms.get_cell()
#         confs = []
#         ef = self.ef
#
#         for a in np.arange(n):
#             indices, offsets = nl.get_neighbors(a)
#             offsets = np.dot(offsets, cell)
#             conf = np.zeros((len(indices), 3))
#             for i, (a2, offset) in enumerate(zip(indices, offsets)):
#                 d = atoms.positions[a2] + offset - atoms.positions[a]
#                 conf[i] = d
#             confs.append(conf)
#
#         confs = np.array(confs)
#
#         ### MULTIPROCESSING ###
#
#         if False:  ### Using the remapped energy ###
#             nods = 2
#             pool = ProcessingPool(nodes=nods)
#             splitind = np.zeros(nods + 1)
#             factor = (n + (nods - 1)) / nods
#             splitind[1:-1] = [(i + 1) * factor for i in np.arange(nods - 1)]
#             splitind[-1] = n
#             splitind = splitind.astype(int)
#             clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nods)]
#             result_energy = np.array(pool.map(ef, clist))
#             result_energy = np.concatenate(result_energy)
#
#         else:
#             result_energy = ef(confs)
#
#         energies = np.reshape(result_energy, n)
#         energy = -np.sum(energies)
#         return energy
#
#     def get_forces(self, atoms):
#
#         ff = self.ff
#
#         n = len(atoms)
#         nl = self.nl
#         nl.update(atoms)
#         cell = atoms.get_cell()
#         confs = []
#
#         for a in np.arange(n):
#             indices, offsets = nl.get_neighbors(a)
#             offsets = np.dot(offsets, cell)
#             conf = np.zeros((len(indices), 3))
#             for i, (a2, offset) in enumerate(zip(indices, offsets)):
#                 d = atoms.positions[a2] + offset - atoms.positions[a]
#                 conf[i] = d
#             confs.append(conf)
#
#         confs = np.array(confs)
#
#         ### MULTIPROCESSING ###
#
#         if False:  ### Using the remapped energy ###
#             nods = 2
#             pool = ProcessingPool(nodes=nods)
#             splitind = np.zeros(nods + 1)
#             factor = (n + (nods - 1)) / nods
#             splitind[1:-1] = [(i + 1) * factor for i in np.arange(nods - 1)]
#             splitind[-1] = n
#             splitind = splitind.astype(int)
#             clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nods)]
#             result_force = np.array(pool.map(ff, clist))
#             result_force = np.concatenate(result_force)
#
#         else:
#             result_force = ff(confs)
#
#         forces = np.reshape(result_force, (n, 3))
#         return forces
#
