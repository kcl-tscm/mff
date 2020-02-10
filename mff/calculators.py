# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod
from itertools import combinations_with_replacement, islice
from pathlib import Path

import numpy as np

from asap3 import FullNeighborList
from ase.calculators.calculator import Calculator, all_changes

logger = logging.getLogger(__name__)


def eam_descriptor(dist, norm, rc, alpha, r0):
    """ Function used to return the eam descriptor for an atom given
        the descriptor's hyperparameters and the set of distances of neighbours.

        Args:
            dist (array): Array of distances of neighbours
            norm (array): Array of versors of neighbours
            rc (float): cutoff radius
            alpha (float): exponent prefactor of the descriptor
            r0 (float): radius in the descriptor

        Returns:
            t2 (float): Eam descriptor 
    """

    q2 = 0.5*(1 + np.cos(np.pi*dist/rc))
    q1 = np.exp(-2*alpha*(dist/r0 - 1))
    q = -sum(q1*q2)**0.5

    dq1 = -2*alpha/r0 * q1

    dq2 = - np.pi/(2*rc) * np.sin(np.pi*dist/rc)

    try:
        dqdrij = -1/(2*q) * (dq1*q2 + q1*dq2)
    except ZeroDivisionError:
        dqdrij = np.zeros(len(q1))
    dqdr = -dqdrij[:, None]*norm
    return q, dqdr


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


class ManySpeciesMappedPotential(Calculator, metaclass=ABCMeta):
    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    @abstractmethod
    def __init__(self, r_cut, elements, **kwargs):
        super().__init__(**kwargs)
        self.elements = elements
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
        results(dict): energy and forces calculated on the atoms object

    """

    def __init__(self, r_cut, grid_2b,  **kwargs):
        """
        Args:
            grid_2b (object): 1D Spline interpolator for the 2-body mapped grid
            r_cut (float): cutoff radius

        """
        super().__init__(r_cut, **kwargs)

        self.grid_2b = grid_2b

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """
        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        for i in range(len(self.atoms)):
            inds, pos, dists2 = self.nl.get_neighbors(i)

            dist = np.sqrt(dists2)
            norm = pos / dist.reshape(-1, 1)

            energy_local = 0.5*self.grid_2b(dist, nu=0)
            fs_scalars = self.grid_2b(dist, nu=1)

            potential_energies[i] = np.sum(energy_local, axis=0)
            forces[i] = np.sum(norm * fs_scalars.reshape(-1, 1), axis=0)

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
            forces[i] += positions[(i, j)] * dE_ij + \
                positions[(i, k)] * dE_ki  # F = - dE/dx
            forces[j] += positions[(j, k)] * dE_jk + \
                positions[(j, i)] * dE_ij  # F = - dE/dx
            forces[k] += positions[(k, i)] * dE_ki + \
                positions[(k, j)] * dE_jk  # F = - dE/dx

            potential_energies[
                [i, j, k]] += energy   # Energy of an atom is the sum of 1/3 of every triplet it is in
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

            # # Limitation
            # assert len(inds) is len(np.unique(inds)
            #                         ), "There are repetitive indices!\n{}".format(inds)

            # ignoring already visited atoms
            inds, pos, dists2 = inds[inds >
                                     i], pos[inds > i, :], dists2[inds > i]
            dists = np.sqrt(dists2)

            for local_ind, (j, pos_ij, dist_ij) in enumerate(zip(inds, pos, dists)):

                # Caching local displacement vectors
                positions[(i, j)], positions[(j, i)] = pos_ij / \
                    dist_ij, -pos_ij / dist_ij

                for k, dist_ik in islice(zip(inds, dists), local_ind + 1, None):

                    try:
                        jk_ind = list(nl[j]).index(k)
                    except ValueError:
                        continue  # no valid triplet

                    _, _, dists_j = nl.get_neighbors(j)

                    indices.append([i, j, k])
                    distances.append(
                        [dist_ij, np.sqrt(dists_j[jk_ind]), dist_ik])

        return np.array(indices), np.array(distances), positions


class EamSingleSpecies(MappedPotential):
    """A mapped Eam calculator for ase

    Attributes:
        grid_eam (object): 1D Spline interpolator for the eam mapped grid
        results(dict): energy and forces calculated on the atoms object

    """

    def __init__(self, r_cut, grid_eam, alpha, r0, **kwargs):
        """
        Args:
            grid_eam (object): 1D Spline interpolator for the eam mapped grid
            r_cut (float): cutoff radius
            alpha (float): Exponential prefactor of the eam Descriptor
            r0 (float): Radius in the exponent of the eam Descriptor

        """
        super().__init__(r_cut, **kwargs)

        self.grid_eam = grid_eam
        self.alpha = alpha
        self.r0 = r0

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """
        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        for i in range(len(self.atoms)):
            inds, pos, dists2 = self.nl.get_neighbors(i)

            dist = np.sqrt(dists2)
            norm = pos / dist.reshape(-1, 1)

            descriptor, descriptor_der = eam_descriptor(
                dist, norm, self.r_cut, self.alpha, self.r0)

            energy_local = self.grid_eam(descriptor, nu=0)
            fs_scalars = self.grid_eam(descriptor, nu=1)

            potential_energies[i] = np.sum(energy_local, axis=0)
            forces[i] = np.sum(
                descriptor_der * fs_scalars.reshape(-1, 1), axis=0)

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces


class CombinedSingleSpecies(TwoBodySingleSpecies, ThreeBodySingleSpecies):
    def __init__(self, r_cut, grid_2b, grid_3b, **kwargs):
        super().__init__(r_cut, grid_2b=grid_2b, grid_3b=grid_3b, **kwargs)


class TwoThreeEamSingleSpecies(TwoBodySingleSpecies, ThreeBodySingleSpecies, EamSingleSpecies):
    def __init__(self, r_cut, grid_2b, grid_3b, grid_eam, alpha, r0, **kwargs):
        super().__init__(r_cut, grid_2b=grid_2b, grid_3b=grid_3b,
                         grid_eam=grid_eam, alpha=alpha, r0=r0, **kwargs)


class TwoBodyManySpecies(ManySpeciesMappedPotential):
    """A mapped 2-body 2-species calculator for ase

    Attributes:
        elements (list): List of ordered atomic numbers of the mapped two species system.
        grids_2b (dict): contains the three 1D Spline interpolators relative to the 2-body 
            mapped grids for element0-element0, element0-element1 and element1-element1 interactions
        results(dict): energy and forces calculated on the atoms object

    """

    def __init__(self, r_cut, elements, grids_2b, **kwargs):
        """
        Args:
            r_cut (float): cutoff radius
            elements (list): List of ordered atomic numbers of the mapped two species system.
            grids_3b (list): contains the four 3D Spline interpolators relative to the 3-body 
                mapped grids for element0-element0-element0, element0-element0-element1, 
                element0-element1-element1 and element1-element1-element1 interactions.

        """
        super().__init__(r_cut, elements, **kwargs)
        self.elements = list(np.sort(elements))
        self.grids_2b = grids_2b

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """

        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        for i, atom in enumerate(self.atoms):
            inds, pos, dists2 = self.nl.get_neighbors(i)

            dist = np.sqrt(dists2)
            norm = pos / dist.reshape(-1, 1)

            energy_local = np.zeros_like(dist)
            fs_scalars = np.zeros_like(dist)

            atom_element_index = atom.number

            for element in self.elements:
                local_inds = np.argwhere(atoms.numbers[inds] == element)
                if len(local_inds) > 0:
                    # Doing this so that the order of the elements is always increasing
                    ellist = (sorted([atom_element_index, element])[0],
                              sorted([atom_element_index, element])[1])
                    local_grid = self.grids_2b[ellist]
                    energy_local[local_inds] = local_grid(
                        dist[local_inds], nu=0)
                    fs_scalars[local_inds] = local_grid(dist[local_inds], nu=1)

            potential_energies[i] = + np.sum(energy_local, axis=0)
            forces[i] = + np.sum(norm * fs_scalars.reshape(-1, 1), axis=0)

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces


class ThreeBodyManySpecies(ManySpeciesMappedPotential):
    """A mapped 3-body 2-species calculator for ase

    Attributes:
        elements (list): List of ordered atomic numbers of the mapped two species system.
        grids_3b (dict): contains the four 3D Spline interpolators relative to the 3-body 
            mapped grids for element0-element0-element0, element0-element0-element1, 
            element0-element1-element1 and element1-element1-element1 interactions.
        results(dict): energy and forces calculated on the atoms object

    """

    def __init__(self, r_cut, elements, grids_3b, **kwargs):
        """
        Args:
            r_cut (float): cutoff radius
            elements (list): List of ordered atomic numbers of the mapped two species system.
            grids_3b (list): contains the four 3D Spline interpolators relative to the 3-body 
                mapped grids for element0-element0-element0, element0-element0-element1, 
                element0-element1-element1 and element1-element1-element1 interactions.

        """
        super().__init__(r_cut, elements, **kwargs)

        self.elements = list(np.sort(elements))
        self.grids_3b = grids_3b

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """
        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))

        indices, distances, positions = self.find_triplets(atoms)

        # Get an array which is a copy of the indices of atoms participating in each triplet
        el_mapping = atoms.get_atomic_numbers()
        el_indices = el_mapping[indices]
        el_indices = np.sort(el_indices, axis=1)

        d_ij, d_jk, d_ki = np.hsplit(distances, 3)

        list_triplets, list_grids = [], []
        perm_list = list(combinations_with_replacement(self.elements, 3))

        for trip in perm_list:
            is_this_the_right_triplet = np.sum(
                el_indices == [trip], axis=1) == 3
            list_triplets.append(is_this_the_right_triplet)
            list_grids.append((trip))

        for r in np.arange(len(list_triplets)):
            mapped = self.grids_3b[list_grids[r]].ev_all(d_ij[list_triplets[r]], d_jk[list_triplets[r]],
                                                         d_ki[list_triplets[r]])
            for (i, j, k), energy, dE_ij, dE_jk, dE_ki in zip(indices[list_triplets[r]], mapped[0], mapped[1],
                                                              mapped[2], mapped[3]):
                forces[i] += positions[(i, j)] * dE_ij + \
                    positions[(i, k)] * dE_ki  # F = - dE/dx
                forces[j] += positions[(j, k)] * dE_jk + \
                    positions[(j, i)] * dE_ij  # F = - dE/dx
                forces[k] += positions[(k, i)] * dE_ki + \
                    positions[(k, j)] * dE_jk  # F = - dE/dx
                potential_energies[
                    [i, j, k]] += energy   # Energy of an atom is the sum of 1/3 of every triplet it is in
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

        indices, distances, positions = [], [], dict()

        for i in range(len(atoms)):

            inds, pos, dists2 = nl.get_neighbors(i)
            # Limitation
            assert len(inds) is len(np.unique(inds)
                                    ), "There are repetitive indices!\n{}".format(inds)

            # ignoring already visited atoms
            inds, pos, dists2 = inds[inds >
                                     i], pos[inds > i, :], dists2[inds > i]
            dists = np.sqrt(dists2)

            for local_ind, (j, pos_ij, dist_ij) in enumerate(zip(inds, pos, dists)):

                # Caching local displacement vectors
                positions[(i, j)], positions[(j, i)] = pos_ij / \
                    dist_ij, -pos_ij / dist_ij

                for k, dist_ik in islice(zip(inds, dists), local_ind + 1, None):

                    try:
                        jk_ind = list(nl[j]).index(k)
                    except ValueError:
                        continue  # no valid triplet

                    _, _, dists_j = nl.get_neighbors(j)

                    indices.append([i, j, k])
                    distances.append(
                        [dist_ij, np.sqrt(dists_j[jk_ind]), dist_ik])

        return np.array(indices), np.array(distances), positions


class EamManySpecies(ManySpeciesMappedPotential):
    """A mapped Eam calculator for ase

    Attributes:
        grid_eam (object): list of 1D Spline interpolator, one for each element in elements
        results(dict): energy and forces calculated on the atoms object

    """

    def __init__(self, r_cut, elements, grids_eam, alpha, r0, **kwargs):
        """
        Args:
            elements (list): List of ordered atomic numbers of the mapped two species system.
            grid_eam (list): list of 1D Spline interpolator, one for each element in elements
            r_cut (float): cutoff radius
            alpha (float): Exponential prefactor of the eam Descriptor
            r0 (float): Radius in the exponent of the eam Descriptor

        """
        super().__init__(r_cut, elements, **kwargs)

        self.elements = list(np.sort(elements))
        self.r_cut = r_cut
        self.grids_eam = grids_eam
        self.alpha = alpha
        self.r0 = r0

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        """Do the calculation.
        """
        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(self.atoms), 3))
        potential_energies = np.zeros((len(self.atoms), 1))
        elements = atoms.get_atomic_numbers()

        for i in range(len(self.atoms)):
            inds, pos, dists2 = self.nl.get_neighbors(i)
            dist = np.sqrt(dists2)
            norm = pos / dist.reshape(-1, 1)

            descriptor, descriptor_der = eam_descriptor(
                dist, norm, self.r_cut, self.alpha, self.r0)

            energy_local = self.grids_eam[elements[i]](descriptor, nu=0)
            fs_scalars = self.grids_eam[elements[i]](descriptor, nu=1)

            potential_energies[i] = np.sum(energy_local, axis=0)
            forces[i] = np.sum(
                descriptor_der * fs_scalars.reshape(-1, 1), axis=0)

        self.results['energy'] += np.sum(potential_energies)
        self.results['forces'] += forces

class CombinedManySpecies(TwoBodyManySpecies, ThreeBodyManySpecies):
    def __init__(self, r_cut, elements, grids_2b, grids_3b, **kwargs):
        super().__init__(r_cut, elements, grids_2b=grids_2b, grids_3b=grids_3b, **kwargs)


class TwoThreeEamManySpecies(TwoBodyManySpecies, ThreeBodyManySpecies, EamManySpecies):
    def __init__(self, r_cut, elements,  grids_2b, grids_3b, grids_eam, alpha, r0, **kwargs):
        super().__init__(r_cut = r_cut, elements= elements, grids_2b=grids_2b, grids_3b=grids_3b,
                         grids_eam=grids_eam, alpha=alpha, r0=r0, **kwargs)

if __name__ == '__main__':
    from ase.io import read
    # from mff.interpolation import Spline3D, Spline1D

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
