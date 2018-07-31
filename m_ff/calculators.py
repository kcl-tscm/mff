# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pathlib import Path
from itertools import islice

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
    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    @abstractmethod
    def __init__(self, r_cut, **kwargs):
        super().__init__(**kwargs)

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

        self.results = {'energy': 0.0,
                        'forces': np.zeros((len(atoms), 3))}

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

    def __init__(self, r_cut, grid_2b, rep_alpha = 0.0, **kwargs):
        super().__init__(r_cut, **kwargs)
        
        self.grid_2b = grid_2b
        self.rep_alpha = rep_alpha
        
    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """

        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))
        
        rep_alpha = self.rep_alpha
        
        for i in range(len(self.atoms)):
            inds, pos, dists2 = self.nl.get_neighbors(i)

            dist = np.sqrt(dists2)
            norm = pos / dist.reshape(-1, 1)

            energy_local = self.grid_2b(dist, nu=0)
            fs_scalars = self.grid_2b(dist, nu=1)

            potential_energies[i] = + 0.5* np.sum(energy_local, axis=0) + 0.5*np.sum((rep_alpha/dist)**12) 
            forces[i] = + np.sum(norm * fs_scalars.reshape(-1, 1), axis=0) - 12*rep_alpha**12*np.einsum('i, in -> n', (1/dist)**13, norm)

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces


class ThreeBodySingleSpecies(MappedPotential):
    """A remapped 3-body calculator for ase
    """

    def __init__(self, r_cut, grid_3b, **kwargs):
        super().__init__(r_cut, **kwargs)

        self.grid_3b = grid_3b

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """

        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        indices, distances, positions = self.find_triplets()

        d_ij, d_jk, d_ki = np.hsplit(distances, 3)

        mapped = self.grid_3b.ev_all(d_ij, d_jk, d_ki)

        for (i, j, k), energy, dE_ij, dE_jk, dE_ki in zip(indices, mapped[0], mapped[1], mapped[2], mapped[3]):            
            forces[i] += positions[(i, j)] * dE_ij + positions[(i, k)] * dE_ki # F = - dE/dx
            forces[j] += positions[(j, k)] * dE_jk + positions[(j, i)] * dE_ij # F = - dE/dx
            forces[k] += positions[(k, i)] * dE_ki + positions[(k, j)] * dE_jk # F = - dE/dx

            potential_energies[
                [i, j, k]] += energy / 3.0  # Energy of an atom is the sum of 1/3 of every triplet it is in

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces

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
            dists = np.sqrt(dists2)

            for local_ind, (j, pos_ij, dist_ij) in enumerate(zip(inds, pos, dists)):

                # Caching local displacement vectors
                positions[(i, j)], positions[(j, i)] = pos_ij / dist_ij, -pos_ij / dist_ij

                for k, dist_ik in islice(zip(inds, dists), local_ind + 1, None):

                    try:
                        jk_ind = list(nl[j]).index(k)
                    except ValueError:
                        continue  # no valid triplet

                    _, _, dists_j = nl.get_neighbors(j)

                    indices.append([i, j, k])
                    distances.append([dist_ij, np.sqrt(dists_j[jk_ind]), dist_ik])

        return np.array(indices), np.array(distances), positions

    # TODO: need to be check
    # def find_triplets2(self):
    #     atoms, nl = self.atoms, self.nl
    #     indices, distances, positions = [], [], dict()
    #
    #     # caching
    #     arr = [nl.get_neighbors(i) for i in range(len(atoms))]
    #
    #     for i, (inds, pos, dists2) in enumerate(arr):
    #         # assert len(inds) is len(np.unique(inds)), "There are repetitive indices!\n{}".format(inds)
    #
    #         # ingnore already visited nodes
    #         inds, pos, dists2 = inds[inds > i], pos[inds > i, :], dists2[inds > i]
    #
    #         dists = np.sqrt(dists2)
    #         pos = pos / dists.reshape(-1, 1)
    #
    #         for (j_ind, j), (k_ind, k) in combinations(enumerate(inds), 2):
    #
    #             jk_ind, = np.where(arr[j][0] == k)
    #
    #             if not jk_ind.size:
    #                 continue  # no valid triplet
    #
    #             indices.append([i, j, k])
    #
    #             # Caching local position vectors
    #             dist_jk = np.sqrt(arr[j][2][jk_ind[0]])
    #             positions[(i, j)], positions[(j, i)] = pos[j_ind], -pos[j_ind]
    #             positions[(i, k)], positions[(k, i)] = pos[k_ind], -pos[k_ind]
    #             positions[(j, k)], positions[(k, j)] = \
    #                 arr[j][1][jk_ind[0], :] / dist_jk, -arr[j][1][jk_ind[0], :] / dist_jk
    #
    #             distances.append([dists[j_ind], dist_jk, dists[k_ind]])
    #
    #     return np.array(indices), np.array(distances), positions


class CombinedSingleSpecies(TwoBodySingleSpecies, ThreeBodySingleSpecies):
    def __init__(self, r_cut, grid_2b, grid_3b, rep_alpha = 0.0, **kwargs):
        super().__init__(r_cut, grid_2b=grid_2b, grid_3b=grid_3b, rep_alpha=rep_alpha, **kwargs)


class TwoBodyTwoSpecies(MappedPotential):
    """A remapped 2-body calculator for ase
    """

    def __init__(self, r_cut, elements, grids_2b, rep_alpha=0.0, **kwargs):
        super().__init__(r_cut, **kwargs)

        self.elements = elements
        self.element_map = {element: index for index, element in enumerate(elements)}
        self.grids_2b = grids_2b
        self.rep_alpha = rep_alpha

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """

        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        rep_alpha = self.rep_alpha
        
        for i, atom in enumerate(self.atoms):
            inds, pos, dists2 = self.nl.get_neighbors(i)

            dist = np.sqrt(dists2)
            norm = pos / dist.reshape(-1, 1)

            energy_local = np.zeros_like(dist)
            fs_scalars = np.zeros_like(dist)

            atom_element_index = self.element_map[atom.number]

            for element in self.elements:
                local_inds = np.argwhere(atoms.numbers[inds] == element)
                if len(local_inds) > 0:
                    # Doing this so that the order of the elements is always increasing
                    ellist = (sorted([atom_element_index, self.element_map[element]])[0], sorted([atom_element_index, self.element_map[element]])[1])
                    local_grid = self.grids_2b[ellist]
                    energy_local[local_inds] = local_grid(dist[local_inds], nu=0)
                    fs_scalars[local_inds] = local_grid(dist[local_inds], nu=1)

            potential_energies[i] = 0.5 * np.sum(energy_local, axis=0) + 0.5*np.sum((rep_alpha / dist) ** 12)
            forces[i] = np.sum(norm * fs_scalars.reshape(-1, 1), axis=0) - 12*rep_alpha**12*np.einsum('i, in -> n', 1/dist**13, norm)
            # forces[i] = np.sum(norm * fs_scalars.reshape(-1, 1), axis=0) - \
            #             12 * rep_alpha ** 12 * np.einsum('i, in -> n', 1 / dist ** 13, norm)

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces


class ThreeBodyTwoSpecies(Calculator):
    pass


if __name__ == '__main__':
    from ase.io import read

    # from m_ff.interpolation import Spline3D, Spline1D

    logging.basicConfig(level=logging.INFO)

    directory = Path('../test/data/Fe_vac')

    filename = directory / 'movie.xyz'
    traj = read(str(filename), index=slice(None))

    # calc = TwoBodySingleSpecies(r_cut=4.45, grid_2b=grid_2b)
    # calc = ThreeBodySingleSpecies(r_cut=4.45, grid_2b=grid_2b, grid_3b=grid_3b)
    #
    # atoms = traj[0]
    # atoms.set_calculator(calc)
    #
    # f = atoms.get_forces()
    # rms = np.sqrt(np.sum(np.square(atoms.arrays['force'] - atoms.get_forces()), axis=1))
    # print('MAEF on forces: {:.4f} +- {:.4f}'.format(np.mean(rms), np.std(rms)))
    #
    # for atoms in traj[0:5]:
    #     atoms.set_calculator(calc)
    #
    #     # print(atoms.arrays['force'])
    #     # print(atoms.get_forces())
    #
    #     rms = np.sqrt(np.sum(np.square(atoms.arrays['force'] - atoms.get_forces()), axis=1))
    #     print('MAEF on forces: {:.4f} +- {:.4f}'.format(np.mean(rms), np.std(rms)))
    #
