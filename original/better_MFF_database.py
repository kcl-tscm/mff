# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 14:55:57 2018

@author: k1506329
"""

import logging

import numpy as np


try:
    from asap3 import FullNeighborList
except:
    from ase.neighborlist import NeighborList

logger = logging.getLogger(__name__)


class MissingData(Exception):
    pass


def carve_from_snapshot(atoms, atoms_ind, r_cut, forces_label=None, energy_label=None, USE_ASAP=True):
    # See if there are forces and energies, get them for the chosen atoms

    forces = atoms.arrays.get(forces_label) if forces_label else atoms.get_forces()
    try:
        energy = atoms.arrays.get(energy_label) if energy_label else atoms.get_potential_energy()
    except:
        energy = None

    if forces is None and energy is None:
        raise MissingData('Cannot find energy or force values in the xyz file, shutting down')

    if forces is not None:
        forces = forces[atoms_ind]
    else:
        logging.info('Forces in the xyz file are not present, or are not called force')

    if energy is None:
        logging.info('Energy in the xyz file is not present, or is not called energy')

    if USE_ASAP:
        # Build local configurations for every indexed atom
        nl = FullNeighborList(r_cut, atoms=atoms)

        confs = []
        for i in atoms_ind:
            indices, positions, distances = nl.get_neighbors(i)

            atomic_numbers_i = np.ones((len(indices), 1)) * atoms.get_atomic_numbers()[i]
            atomic_numbers_j = atoms.get_atomic_numbers()[indices].reshape(-1, 1)

            confs.append(np.hstack([positions, atomic_numbers_i, atomic_numbers_j]))
    else:

        # Build local configurations for every indexed atom
        cutoffs = np.ones(len(atoms)) * r_cut / 2.
        nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False, bothways=True)
        nl.update(atoms)

        confs = []
        cell = atoms.get_cell()

        for i in atoms_ind:
            indices, offsets = nl.get_neighbors(i)
            offsets = np.dot(offsets, cell)
            conf = np.zeros((len(indices), 5))

            for k, (a2, offset) in enumerate(zip(indices, offsets)):
                d = atoms.positions[a2] + offset - atoms.positions[i]
                conf[k, :3] = d
                conf[k, 4] = atoms.get_atomic_numbers()[a2]

            conf[:, 3] = atoms.get_atomic_numbers()[i]
            confs.append(conf)

    return confs, forces, energy


def carve_confs(atoms, r_cut, n_data, forces_label=None, energy_label=None, USE_ASAP=True):
    confs, forces, energies = [], [], []

    # Get the atomic number of each atom in the trajectory file
    atom_number_list = [atom.get_atomic_numbers() for atom in atoms]
    flat_atom_number = np.concatenate(atom_number_list)
    elements, elements_count = np.unique(flat_atom_number, return_counts=True)

    # Calculate the ratios of occurrence of central atoms based on their atomic number
    ratios = np.sqrt(1.0 * elements_count) / np.sum(elements_count)
    ratios /= np.sum(ratios)

    if n_data > np.sum(elements_count):
        print('WARNING: n_data is larger that total number of configuration')
        n_data = np.sum(elements_count)
        print('Settin n_data = {}'.format(n_data))

    # Obtain the indices of the atoms we want in the final database from a linspace on the the flattened array
    indices = [np.linspace(0, elc, int(ratio * n_data) - 1, dtype=np.int) for elc, ratio in zip(elements_count, ratios)]

    # Go through each trajectory step and find where the chosen indexes for all different elements are
    element_ind_count = np.zeros_like(elements)
    element_ind_count_prev = np.zeros_like(elements)

    for j in np.arange(len(atoms)):
        logging.info('Reading traj step {}'.format(j))

        this_ind = []

        for k in np.arange(len(elements)):
            count_el_atoms = sum(atom_number_list[j] == elements[k])
            element_ind_count[k] += count_el_atoms
            temp_ind = np.array([x for x in (indices[k] - element_ind_count_prev[k]) if (0 <= x < count_el_atoms)],
                                dtype=np.int)

            this_ind.append((np.where(atom_number_list[j] == elements[k]))[0][temp_ind])
            element_ind_count_prev[k] += count_el_atoms

        this_ind = np.concatenate(this_ind).ravel()

        # Call the carve_from_snapshot function on the chosen atoms
        if this_ind.size > 0:
            this_conf, this_force, this_energy = \
                carve_from_snapshot(atoms[j], this_ind, r_cut, forces_label, energy_label, USE_ASAP)
            confs.append(this_conf)
            forces.append(this_force)
            energies.append(this_energy)

    # Reshape everything so that confs is a list of numpy arrays, forces is a numpy array and energies is a numpy array
    confs = [item for sublist in confs for item in sublist]
    forces = [item for sublist in forces for item in sublist]

    forces = np.asarray(forces)
    energies = np.asarray(energies)

    return elements, confs, forces, energies


if __name__ == '__main__':
    from ase.io import read

    logging.basicConfig(level=logging.INFO)

    r_cut = 100.0
    n_data = 3000

    # Open file and get number of atoms and steps
    directory = '../test/data/BIP_300/'
    filename = 'movie.xyz'
    traj = read(directory + '/' + filename, index=slice(0, 240), format='extxyz')

    elements, confs, forces, energies = carve_confs(traj, r_cut, n_data, forces_label='force')

    np.save(directory + '/' + 'confs_cut={:.2f}.npy'.format(r_cut), confs)
    np.save(directory + '/' + 'forces_cut={:.2f}.npy'.format(r_cut), forces)
    np.save(directory + '/' + 'energies_cut={:.2f}.npy'.format(r_cut), energies)

    lens = [len(conf) for conf in confs]

    logging.info('\n'.join((
        'Number of atoms in a configuration:',
        '   maximum: {}'.format(np.max(lens)),
        '   minimum: {}'.format(np.min(lens)),
        '   average: {:.4}'.format(np.mean(lens))
    )))
