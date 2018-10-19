# -*- coding: utf-8 -*-
"""
Configurations
==============

Module used to create ase calculators using interpolators
constructed with the models module of the m_ff package.
These calculators can yield forces and energies for atoms objects
at a very low computational cost.

WARNING: The atoms object must be such that the atoms are ordered
with increasing atomic number for the two species calculator to work.

Example:

    >>> calc = calculators.CombinedTwoSpecies(
            r_cut, element0, element1,  grid_2b, grid_3b, rep_alpha=1.4)
    >>> atoms = atoms[np.argsort(atoms.get_atomic_numbers())]
    >>> atoms.set_calculator(calc)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import numpy as np
import logging

from abc import ABCMeta, abstractmethod
from pathlib import Path
from itertools import islice
from ase.calculators.calculator import Calculator, all_changes
from asap3 import FullNeighborList

logger = logging.getLogger(__name__)


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


class TwoSpeciesMappedPotential(Calculator, metaclass=ABCMeta):
    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    @abstractmethod
    def __init__(self, r_cut, element0, element1, **kwargs):
        super().__init__(**kwargs)
        self.element0 = element0
        self.element1 = element1
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
    """A mapped 2-body calculator for ase
   
    Attributes:
        grid_2b (object): 1D Spline interpolator for the 2-body mapped grid
        rep_alpha (float): Repulsion parameter, used when no data for very 
            close atoms are available in order to avoid collisions during MD.
            The parameter governs a repulsion force added to the computed one.
        results(dict): energy and forces calculated on the atoms object
    
    """

    def __init__(self, r_cut, grid_2b, rep_alpha=0.0, **kwargs):
        """
        Args:
            grid_2b (object): 1D Spline interpolator for the 2-body mapped grid
            r_cut (float): cutoff radius
            rep_alpha (float): Repulsion parameter, used when no data for very 
                close atoms are available in order to avoid collisions during MD.
                The parameter governs a repulsion force added to the computed one.
            
        """
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

            # The energy and force from the repulsion potential governed by rep_alpha are added here
            potential_energies[i] = (+ 0.5 * np.sum(energy_local, axis=0) +
                                     0.5 * np.sum((rep_alpha / dist) ** 12))
            forces[i] = (+ np.sum(norm * fs_scalars.reshape(-1, 1), axis=0) -
                         12 * rep_alpha ** 12 * np.einsum('i, in -> n', (1 / dist) ** 13, norm))

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces


class ThreeBodySingleSpecies(MappedPotential):
    """A mapped 3-body calculator for ase
   
    Attributes:
        grid_3b (object): 3D Spline interpolator for the 3-body mapped grid
        results(dict): energy and forces calculated on the atoms object
    
    """

    def __init__(self, r_cut, grid_3b, **kwargs):
        """
        Args:
            grid_3b (object): 3D Spline interpolator for the 3-body mapped grid
            r_cut (float): cutoff radius
            
        """
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
            forces[i] += positions[(i, j)] * dE_ij + positions[(i, k)] * dE_ki  # F = - dE/dx
            forces[j] += positions[(j, k)] * dE_jk + positions[(j, i)] * dE_ij  # F = - dE/dx
            forces[k] += positions[(k, i)] * dE_ki + positions[(k, j)] * dE_jk  # F = - dE/dx

            potential_energies[
                [i, j, k]] += energy / 3.0  # Energy of an atom is the sum of 1/3 of every triplet it is in
        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces

    def find_triplets(self):
        '''Function that efficiently finds all of the valid triplets of atoms in the atoms object.
        
        Returns:
            indices (array): array containing the indices of atoms belonging to any valid triplet.
                Has shape T by 3 where T is the number of valid triplets in the atoms object
            distances (array): array containing the relative distances of every triplet of atoms.
                Has shape T by 3 where T is the number of valid triplets in the atoms object
            positions (dictionary): versor of position w.r.t. the central atom of every atom indexed in indices.
                Has shape T by 3 where T is the number of valid triplets in the atoms object            
        
        '''
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

    # TODO: need to be checked
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
    def __init__(self, r_cut, grid_2b, grid_3b, rep_alpha=0.0, **kwargs):
        super().__init__(r_cut, grid_2b=grid_2b, grid_3b=grid_3b, rep_alpha=rep_alpha, **kwargs)


class TwoBodyTwoSpecies(TwoSpeciesMappedPotential):
    """A mapped 2-body 2-species calculator for ase
   
    Attributes:
        elements (list): List of ordered atomic numbers of the mapped two species system.
        grids_2b (dict): contains the three 1D Spline interpolators relative to the 2-body 
            mapped grids for element0-element0, element0-element1 and element1-element1 interactions
        rep_alpha (float): Repulsion parameter, used when no data for very 
            close atoms are available in order to avoid collisions during MD.
            The parameter governs a repulsion force added to the computed one.
        results(dict): energy and forces calculated on the atoms object
    
    """

    def __init__(self, r_cut, element0, element1, grids_2b, rep_alpha=0.0, **kwargs):
        """
        Args:
            r_cut (float): cutoff radius
            element0 (int): atomic number of the first element in ascending order
            element1 (int): atomic number of the second element in ascending order
            grids_3b (list): contains the four 3D Spline interpolators relative to the 3-body 
                mapped grids for element0-element0-element0, element0-element0-element1, 
                element0-element1-element1 and element1-element1-element1 interactions.
            rep_alpha (float): Repulsion parameter, used when no data for very 
            close atoms are available in order to avoid collisions during MD.
            The parameter governs a repulsion force added to the computed one.
        
        """
        super().__init__(r_cut, element0, element1, **kwargs)
        elements = [element0, element1]
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
                    ellist = (sorted([atom_element_index, self.element_map[element]])[0],
                              sorted([atom_element_index, self.element_map[element]])[1])
                    local_grid = self.grids_2b[ellist]
                    energy_local[local_inds] = local_grid(dist[local_inds], nu=0)
                    fs_scalars[local_inds] = local_grid(dist[local_inds], nu=1)

            potential_energies[i] = + 0.5 * np.sum(energy_local, axis=0) + 0.5 * np.sum((rep_alpha / dist) ** 12)
            forces[i] = + np.sum(norm * fs_scalars.reshape(-1, 1), axis=0) - 12 * rep_alpha ** 12 * np.einsum(
                'i, in -> n', (1 / dist) ** 13, norm)

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces


class ThreeBodyTwoSpecies(TwoSpeciesMappedPotential):
    """A mapped 3-body 2-species calculator for ase
   
    Attributes:
        elements (list): List of ordered atomic numbers of the mapped two species system.
        grids_3b (dict): contains the four 3D Spline interpolators relative to the 3-body 
            mapped grids for element0-element0-element0, element0-element0-element1, 
            element0-element1-element1 and element1-element1-element1 interactions.
        results(dict): energy and forces calculated on the atoms object
    
    """

    def __init__(self, r_cut, element0, element1, grids_3b, **kwargs):
        """
        Args:
            r_cut (float): cutoff radius
            element0 (int): atomic number of the first element in ascending order
            element1 (int): atomic number of the second element in ascending order
            grids_3b (list): contains the four 3D Spline interpolators relative to the 3-body 
                mapped grids for element0-element0-element0, element0-element0-element1, 
                element0-element1-element1 and element1-element1-element1 interactions.
        
        """
        super().__init__(r_cut, element0, element1, **kwargs)
        elements = [element0, element1]

        self.elements = elements
        self.element_map = {element: index for index, element in enumerate(elements)}
        self.grids_3b = grids_3b

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """
        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        indices, distances, positions = self.find_triplets(atoms)

        el_indices = indices.copy()
        element_list = atoms.get_atomic_numbers()
        # Find the index of the last element0 atom
        n_first_element = (element_list == self.elements[0]).sum()

        # Indixes of triplets that have 0 or 1 depending on the element of each participating atom
        el_indices[el_indices < n_first_element] = 0
        el_indices[el_indices > 0] = 1

        d_ij, d_jk, d_ki = np.hsplit(distances, 3)
        list_000 = (1 - (np.sum(1 - el_indices == [0, 0, 0], axis=1)).astype(bool)).astype(bool)
        list_001 = (1 - (np.sum(1 - el_indices == [0, 0, 1], axis=1)).astype(bool)).astype(bool)
        list_011 = (1 - (np.sum(1 - el_indices == [0, 1, 1], axis=1)).astype(bool)).astype(bool)
        list_111 = (1 - (np.sum(1 - el_indices == [1, 1, 1], axis=1)).astype(bool)).astype(bool)
        list_triplets = [list_000, list_001, list_011, list_111]
        list_grids = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

        for r in np.arange(4):
            mapped = self.grids_3b[list_grids[r]].ev_all(d_ij[list_triplets[r]], d_jk[list_triplets[r]],
                                                         d_ki[list_triplets[r]])
            for (i, j, k), energy, dE_ij, dE_jk, dE_ki in zip(indices[list_triplets[r]], mapped[0], mapped[1],
                                                              mapped[2], mapped[3]):
                forces[i] += positions[(i, j)] * dE_ij + positions[(i, k)] * dE_ki  # F = - dE/dx
                forces[j] += positions[(j, k)] * dE_jk + positions[(j, i)] * dE_ij  # F = - dE/dx
                forces[k] += positions[(k, i)] * dE_ki + positions[(k, j)] * dE_jk  # F = - dE/dx
                potential_energies[
                    [i, j, k]] += energy / 3.0  # Energy of an atom is the sum of 1/3 of every triplet it is in
        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces

    def find_triplets(self, atoms):
        '''Function that efficiently finds all of the valid triplets of atoms in the atoms object.
        
        Returns:
            indices (array): array containing the indices of atoms belonging to any valid triplet.
                Has shape T by 3 where T is the number of valid triplets in the atoms object
            distances (array): array containing the relative distances of every triplet of atoms.
                Has shape T by 3 where T is the number of valid triplets in the atoms object
            positions (dictionary): versor of position w.r.t. the central atom of every atom indexed in indices.
                Has shape T by 3 where T is the number of valid triplets in the atoms object            
        
        '''
        nl = self.nl
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


class CombinedTwoSpecies(TwoBodyTwoSpecies, ThreeBodyTwoSpecies):
    def __init__(self, r_cut, element0, element1, grids_2b, grids_3b, rep_alpha=0.0, **kwargs):
        super().__init__(r_cut, element0, element1, grids_2b=grids_2b, grids_3b=grids_3b, rep_alpha=rep_alpha, **kwargs)


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
