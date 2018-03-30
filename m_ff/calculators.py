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

from abc import ABCMeta, abstractmethod
from pathlib import Path
from itertools import combinations, islice

import numpy as np
import logging

from ase.calculators.calculator import Calculator, all_changes
from asap3 import FullNeighborList

logger = logging.getLogger(__name__)


# TODO: defining the final/proper grid object
# TODO: testing basic operation on single species 2 and 3 body
# TODO: implementation of two species (idea: sum of element index)
# TODO: Factory method

class SingleSpecies(Exception):
    pass


class MappedPotential(Calculator, metaclass=ABCMeta):
    @abstractmethod
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


class TwoBodySingleSpecies(MappedPotential):
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

        # TODO: proper numpy itereator with enumaration
        for i in range(len(self.atoms)):
            inds, pos, dists2 = self.nl.get_neighbors(i)

            energy_local = self.grid_1_1(np.sqrt(dists2), nu=0)
            fs_scalars = - self.grid_1_1(np.sqrt(dists2), nu=1)

            potential_energies[i] = np.sum(energy_local, axis=0)
            forces[i] = np.sum(pos * fs_scalars[:, np.newaxis], axis=0)

        self.results = {'energy': np.sum(potential_energies),
                        'forces': forces}


class ThreeBodySingleSpecies(MappedPotential):
    """A remapped 3-body calculator for ase
    """

    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    def __init__(self, r_cut, grid_1_1_1, restart=None, ignore_bad_restart_file=False, label=None, atoms=None,
                 **kwargs):
        super().__init__(r_cut, restart, ignore_bad_restart_file, label, atoms, **kwargs)

        self.grid_1_1_1 = grid_1_1_1

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

        indices, distances, positions = self.find_triplets()
        xi, yi, zi = np.hsplit(distances, 3)
        mapped = self.grid_1_1_1.ev_all(xi, yi, zi)

        for (i, j, k), energy, dE_ij, dE_jk, dE_ki in zip(indices, mapped[0], mapped[1], mapped[2], mapped[3]):
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

            inds, pos, dists2 = nl.get_neighbors(i)

            # Limitation
            assert len(inds) is len(np.unique(inds)), "There are repetitive indices!\n{}".format(inds)

            # ignoring already visited atoms
            inds, pos, dists2 = inds[inds > i], pos[inds > i, :], dists2[inds > i]

            for local_ind, (j, pos_ij, dist_ij) in enumerate(zip(inds, pos, dists2)):

                # Caching local displacement vectors
                positions[(i, j)], positions[(j, i)] = pos_ij, -pos_ij

                for k, dist_ik in islice(zip(inds, dists2), local_ind + 1, None):

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


class TwoBodyTwoSpecies(Calculator):
    pass


class ThreeBodyTwoSpecies(Calculator):
    pass


if __name__ == '__main__':
    from ase.io import read
    from m_ff.interpolation import Spline3D

    logging.basicConfig(level=logging.INFO)

    directory = Path('../test/data/Fe_vac')

    filename = directory / 'movie.xyz'
    traj = read(str(filename), index=slice(None))

    rs, element1, _, _, grid_1_1_1 = np.load(str(directory / 'MFF_3b_ntr_10_sig_1.00_cut_4.45.npy'))
    grid_1_1_1 = Spline3D(rs, rs, rs, grid_1_1_1)

    calc = ThreeBodySingleSpecies(r_cut=3.7, grid_1_1_1=grid_1_1_1)

    atoms = traj[0]
    atoms.set_calculator(calc)

    f = atoms.get_forces()
    rms = np.sqrt(np.sum(np.square(atoms.arrays['force'] - atoms.get_forces()), axis=1))
    print('MAEF on forces: {:.4f} +- {:.4f}'.format(np.mean(rms), np.std(rms)))

    # for atoms in traj:
    #     atoms.set_calculator(calc)
    #
    #     rms = np.sqrt(np.sum(np.square(atoms.arrays['force'] - atoms.get_forces()), axis=1))
    #     print('MAEF on forces: {:.4f} +- {:.4f}'.format(np.mean(rms), np.std(rms)))
    #
    # pass

# If single element, build only a 2- and  3-body grid
# result = [rs, element1, element2, grid_1_1, grid_1_1_1]

# If two elements, build three 2- and four 3-body grids
# result = [rs, element1, element2, grid_1_1, grid_1_2, grid_2_2, grid_1_1_1, grid_1_1_2, grid_1_2_2, grid_2_2_2]



