# -*- coding: utf-8 -*-
"""
Configurations
==============

Module used to extract local atomic environments
from ase atoms objects.

Example:

    >>> trajectory = ase.io.read(filename)
    >>> elements, confs, forces, energies = configurations.carve_confs(
                trajcetory, r_cut, n_data)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import logging
import numpy as np

from abc import ABCMeta, abstractmethod
from asap3 import FullNeighborList

logger = logging.getLogger(__name__)


class MissingData(Exception):
    pass


def carve_from_snapshot(atoms, atoms_ind, r_cut, forces_label=None, energy_label=None):
    """Extract atomic configurations, the forces acting on the central atoms
    os said configurations, and the local energy values associated to a single atoms object.

    Args:
        atoms (ase atoms object): Ase atoms file, opened with ase.io.read
        atoms_ind (list): indexes of the atoms for which a conf is created
        r_cut (float): Cutoff to use when carving out atomic environments
        forces_label (str): Name of the force label in the trajectory file, if None default is "forces"
        energy_label (str): Name of the energy label in the trajectory file, if None default is "energy"
        
    Returns:
        confs (list of arrays): List of M by 5 numpy arrays, where M is the number of atoms within
            r_cut from the central one. The first 3 components are positions w.r.t
            the central atom in Angstroms, the fourth is the atomic number of the 
            central atom, the fifth the atomic number of each atom.
        forces (array): x,y,z components of the force acting on the central atom in eV/Angstrom
        energies (array): value of the local atomic energy in eV

    """

    # See if there are forces and energies, get them for the chosen atoms
    if (atoms.get_cell() == np.zeros((3, 3))).all():
        atoms.set_cell(100.0 * np.identity(3))
        logger.warning('No cell values found, setting to a 100 x 100 x 100 cube')

    if forces_label:
        forces = atoms.arrays.get(forces_label)
    else:
        forces = atoms.get_forces()
        forces_label = 'forces'

    if energy_label and energy_label != 'energy':
        energy = atoms.arrays.get(energy_label)
    else:
        energy_label = 'energy'
        try:
            energy = atoms.get_potential_energy()
        except:
            energy = None

    if forces is None and energy is None:
        raise MissingData('Cannot find energy or force values in the xyz file, shutting down')

    if forces is not None:
        forces = forces[atoms_ind]
    else:
        logger.info('Forces in the xyz file are not present, or are not called %s' % (forces_label))

    if energy is None:
        logger.info('Energy in the xyz file is not present, or is not called %s' % (energy_label))

    # Build local configurations for every indexed atom
    nl = FullNeighborList(r_cut, atoms=atoms)
    confs = []
    for i in atoms_ind:
        indices, positions, distances = nl.get_neighbors(i)

        atomic_numbers_i = np.ones((len(indices), 1)) * atoms.get_atomic_numbers()[i]
        atomic_numbers_j = atoms.get_atomic_numbers()[indices].reshape(-1, 1)

        confs.append(np.hstack([positions, atomic_numbers_i, atomic_numbers_j]))
    return confs, forces, energy


def carve_confs(atoms, r_cut, n_data, forces_label=None, energy_label=None, boundaries=None):
    """Extract atomic configurations, the forces acting on the central atoms
    os said configurations, and the local energy values associeated.

    Args:
        atoms (ase atoms object): Ase trajectory file, opened with ase.io.read
        r_cut (float): Cutoff to use when carving out atomic environments
        n_data (int): Total number of atomic configurations to extract from the trajectory
        forces_label (str): Name of the force label in the trajectory file, if None default is "forces"
        energy_label (str): Name of the energy label in the trajectory file, if None default is "energy"
        boundaries (list): List containing three lists for the three cartesian coordinates. 
            Each of them contains a list of tuples indicating every interval that must be used to
            sample central atoms from. Example: boundaries = [[], [], [[-10.0, +5.3]]]
            Default is None, and all of the snapshot is used.

    Returns:
        elements (array): Array of atomic numbers in increasing order
        confs (list of arrays): List of M by 5 numpy arrays, where M is the number of atoms within
            r_cut from the central one. The first 3 components are positions w.r.t
            the central atom in Angstroms, the fourth is the atomic number of the 
            central atom, the fifth the atomic number of each atom.
        forces (array): x,y,z components of the force acting on the central atom in eV/Angstrom
        energies (array): value of the local atomic energy in eV

    """

    confs, forces, energies = [], [], []

    # Get the atomic number of each atom in the trajectory file
    atom_number_list = [atom.get_atomic_numbers() for atom in atoms]
    flat_atom_number = np.concatenate(atom_number_list)
    elements, elements_count = np.unique(flat_atom_number, return_counts=True)

    # Calculate the ratios of occurrence of central atoms based on their atomic number
    if n_data > np.sum(elements_count):
        logger.warning('n_data is larger that total number of configuration')
        n_data = np.sum(elements_count)
        print('Setting n_data = {}'.format(n_data))

    # Obtain the indices of the atoms we want in the final database from a linspace on the the flattened array
    if len(elements) > 1:
        ratios = np.sqrt(1.0 * elements_count) / np.sum(elements_count)
        ratios /= np.sum(ratios)

    # if smart_sampling:
    #     indices = [np.linspace(0, elc, int(ratio * n_data) - 1, dtype=np.int) for elc, ratio in zip(elements_count, ratios)]
    # else:

    avg_natoms = len(flat_atom_number) // len(atoms)
    # Use randomly selected snapshots to roughly match n_data
    ind_snapshot = np.random.randint(0, len(atoms) - 1, n_data // avg_natoms)

    # Go through each trajectory step and find where the chosen indexes for all different elements are
    element_ind_count = np.zeros_like(elements)
    element_ind_count_prev = np.zeros_like(elements)

    for j in np.arange(len(atoms)):
        this_ind = []
        logger.info('Reading traj step {}'.format(j))

        # if smart_sampling:
        #     for k in np.arange(len(elements)):
        #         count_el_atoms = sum(atom_number_list[j] == elements[k])
        #         element_ind_count[k] += count_el_atoms
        #         temp_ind = np.array([x for x in (indices[k] - element_ind_count_prev[k]) if (0 <= x < count_el_atoms)],
        #                             dtype=np.int)
        #
        #         this_ind.append((np.where(atom_number_list[j] == elements[k]))[0][temp_ind])
        #         element_ind_count_prev[k] += count_el_atoms
        #
        #     this_ind = np.concatenate(this_ind).ravel()
        # else:

        if j in ind_snapshot:  # For every snapshot, if that was selected through random sampling, do the following
            positions = atoms[j].get_positions()
            if boundaries is not None:  # Select only atoms within the boundaries
                for d in np.arange(3):
                    for i in np.arange(len(boundaries[d])):
                        this_ind.append(np.ravel(np.argwhere(np.logical_and(
                            positions[:, d] > boundaries[d][i][0], positions[:, d] < boundaries[d][i][1]))))
            else:
                this_ind.append(np.asarray(np.arange(len(atoms[j]))))
            this_ind = np.concatenate(this_ind).ravel()

        # Call the carve_from_snapshot function on the chosen atoms
        if len(this_ind) > 0:
            this_conf, this_force, this_energy = \
                carve_from_snapshot(atoms[j], this_ind, r_cut, forces_label, energy_label)
            confs.append(this_conf)
            forces.append(this_force)
            energies.append(this_energy)

    # Reshape everything so that confs is a list of numpy arrays, forces is a numpy array and energies is a numpy array
    confs = [item for sublist in confs for item in sublist]
    forces = [item for sublist in forces for item in sublist]
    forces = np.asarray(forces)
    energies = np.asarray(energies)

    if boundaries is not None:
        logger.warning('When using boundaries, n_data is smaller than the one set by user')

    return elements, confs, forces, energies


class Configurations(metaclass=ABCMeta):
    """Configurations can represent a list of configurations"""

    @abstractmethod
    def __init__(self, confs=None):
        self.confs = confs if confs else []

    def save(self, *args):
        pass


class SingleSpecies(metaclass=ABCMeta):
    pass


class MultiSpecies(metaclass=ABCMeta):
    pass


class Forces(Configurations, metaclass=ABCMeta):
    def __init__(self, confs=None, forces=None):
        super().__init__(confs)
        self.forces = forces if forces else []


class Energies(Configurations, metaclass=ABCMeta):
    def __init__(self, confs=None, energy=None):
        super().__init__(confs)
        self.energy = energy if energy else []


class ConfsTwoBodySingleForces(Forces, SingleSpecies):
    def __init__(self, r_cut):
        super().__init__()

        self.r_cut = r_cut

    def append(self, atoms, atom_inds=None, forces_label=None):

        if atom_inds is None:
            atom_inds = range(len(atoms))

        # See if there are forces and energies, get them for the chosen atoms
        # forces = atoms.arrays.get(forces_label)
        forces = atoms.arrays.get(forces_label) if forces_label else atoms.get_forces()

        if forces is None:
            raise MissingData('Forces in the Atoms objects under {} label are not present.'.format(forces_label))

        self.forces.append(forces[atom_inds])

        # Build local configurations for every indexed atom
        nl = FullNeighborList(self.r_cut, atoms=atoms)

        for i in atom_inds:
            indices, positions, distances = nl.get_neighbors(i)
            self.confs.append(positions)

    def save(self, filename_conf, filename_force):
        np.save(filename_conf, self.confs)
        np.save(filename_force, self.forces)


class ConfsTwoForces(Forces, SingleSpecies):
    def __init__(self, r_cut):
        super().__init__()

        self.r_cut = r_cut

    def append(self, atoms, atom_inds=None, forces_label='force'):

        if atom_inds is None:
            atom_inds = range(len(atoms))

        # See if there are forces and energies, get them for the chosen atoms
        forces = atoms.arrays.get(forces_label)

        if forces is None:
            raise MissingData('Forces in the Atoms objects under {} label are not present.'.format(forces_label))

        self.forces.append(forces[atom_inds])

        # Build local configurations for every indexed atom
        nl = FullNeighborList(self.r_cut, atoms=atoms)

        for i in atom_inds:
            indices, positions, distances = nl.get_neighbors(i)
            self.confs.append(positions)

    def save(self, filename_conf, filename_force):
        np.save(filename_conf, self.confs)
        np.save(filename_force, self.forces)


if __name__ == '__main__':
    from ase.io import read
    from mff.sampling import SamplingRandomSingle

    testfiles = {
        'BIP_300': '../test/data/BIP_300/movie.xyz',
        'C_a': '../test/data/C_a/data_C.xyz',
        'Fe_vac': '../test/data/Fe_vac/movie.xyz',
        'HNi': '../test/data/HNI/h_ase500.xyz'
    }

    filename = testfiles['Fe_vac']
    traj = read(filename, index=slice(0, 4))
    print(traj)

    confs = ConfsTwoBodySingleForces(r_cut=4.5)
    for atoms, atom_inds in SamplingRandomSingle(traj, n_target=25):
        confs.append(atoms, atom_inds)


# if __name__ == '__main__':
#
#     from ase.io import read, iread
#
#     filename = '../data-lammps/Silic_300/Si_300_dump.atom'
#
#     r_cut = 4.17
#
#     confs_list = []
#     force_list = []
#
#     # slice(start, stop, increment)
#     for atoms in iread(filename, index=slice(None), format='lammps-dump'):
#
#         # Quick fixes:
#         # ============
#         # pbc
#         atoms.set_pbc([True, True, True])
#         # mapping atoms types to atomic number (Si: 1 -> 14)
#         atom_numbers = atoms.get_atomic_numbers()
#         np.place(atom_numbers, atom_numbers == 1, 14)
#         atoms.set_atomic_numbers(atom_numbers)
#
#         forces = atoms.calc.results['forces']
#
#         # Testing (64k confs):
#         # confs1 = list(get_confs(atoms, r_cut)) # 32 sec 1x
#         # confs2 = list(get_confs_asap(atoms, r_cut)) # 0.6 sec 50x
#         # confs3 = list(get_confs_cKDTree(atoms, r_cut)) # 2 sec 16x
#
#         for conf, force in zip(get_confs_asap(atoms, r_cut), forces):
#             confs_list.append(conf)
#             force_list.append(force)
