# -*- coding: utf-8 -*-

import json
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from random import shuffle

import numpy as np
from scipy.spatial.distance import cdist

from asap3 import FullNeighborList
from ase.io import read

logger = logging.getLogger(__name__)


class MissingData(Exception):
    pass


def carve_from_snapshot(atoms, r_cut, forces_label=None, energy_label=None, atoms_ind=None):
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

    if atoms_ind is None:
        atoms_ind = np.arange(len(atoms))

    if forces_label:
        forces = atoms.arrays.get(forces_label)
    else:
        try:
            forces = atoms.get_forces()
        except:
            forces = None
    if energy_label and energy_label != 'energy':
        energy = atoms.arrays.get(energy_label)
    else:
        energy_label = 'energy'
        try:
            energy = atoms.get_potential_energy()
        except:
            energy = None

    if forces is None and energy is None:
        raise MissingData(
            'Cannot find energy or force values in the xyz file, shutting down')

    if forces is not None:
        forces = forces[atoms_ind]
    else:
        logger.info(
            'Forces in the xyz file are not present, or are not called %s' % (forces_label))

    if energy is None:
        logger.info(
            'Energy in the xyz file is not present, or is not called %s' % (energy_label))

    # See if there are forces and energies, get them for the chosen atoms
    if (atoms.get_cell() == np.zeros((3, 3))).all():
        atoms.set_cell(100.0 * np.identity(3))
        logger.info('No cell values found, setting to a 100 x 100 x 100 cube')

    # Build local configurations for every indexed atom
    nl = FullNeighborList(r_cut, atoms=atoms)
    confs = []
    for i in atoms_ind:
        indices, positions, distances = nl.get_neighbors(i)

        atomic_numbers_i = np.ones(
            (len(indices), 1)) * atoms.get_atomic_numbers()[i]
        atomic_numbers_j = atoms.get_atomic_numbers()[indices].reshape(-1, 1)
        confs.append(
            np.hstack([positions, atomic_numbers_i, atomic_numbers_j]))

    return confs, forces, energy


def generate(traj, r_cut, forces_label=None, energy_label=None):
    """Extract atomic configurations, the forces acting on the central atoms
    os said configurations, and the local energy values associeated.

    Args:
        traj (ase atoms object): Ase trajectory file, opened with ase.io.read
        r_cut (float): Cutoff to use when carving out atomic environments
        forces_label (str): Name of the force label in the trajectory file, if None default is "forces"
        energy_label (str): Name of the energy label in the trajectory file, if None default is "energy"

    Returns:
        data (dictionary): Structure containing, for each snapshot in the trajectory, 
            the forces, energy, and local atomic configurations for that snapshot's atoms

    """

    # Get the atomic number of each atom in the trajectory file
    atom_number_list = [atoms.get_atomic_numbers() for atoms in traj]
    flat_atom_number = np.concatenate(atom_number_list)
    elements = np.unique(flat_atom_number, return_counts=False)

    elements = list(elements)
    data = {}
    data['elements'] = elements
    data['r_cut'] = r_cut
    data['n_steps'] = len(traj)
    data['data'] = []

    for i, atoms in enumerate(traj):
        this_conf, this_force, this_energy = \
            carve_from_snapshot(
                atoms, r_cut, forces_label=forces_label, energy_label=energy_label)
        this_step = {}
        this_step['confs'] = this_conf
        this_step['forces'] = this_force
        this_step['energy'] = this_energy

        data['data'].append(this_step)

    return data


def save(path, r_cut, data):
    """ Save data extracted with ``generate`` to a file with a iven cutoff

    Args:
        path (Path or string): Name and position of file to save data to
        r_cut (float): Cutoff used
        data (dict): Structure containing, for each snapshot in the trajectory, 
            the forces, energy, and local atomic configurations for that snapshot's atoms.
            Obtained from ``generate``

    """

    if not isinstance(path, Path):
        path = Path(path)

    np.save('{}/data_cut={:.2f}.npy'.format(path, r_cut), data)


def generate_and_save(path, r_cut, forces_label=None, energy_label=None, index=':'):
    """ Generate the data dictionary and save it to the same location.
    Args:
        path (Path or string): Name and position of trajectory file
        r_cut (float): Cutoff used
        forces_label (str): Name of the force label in the trajectory file, if None default is "forces"
        energy_label (str): Name of the energy label in the trajectory file, if None default is "energy"
        index (str): Indexes indicating which snapshots to use from the traj file

    Returns:
        data (dict): Structure containing, for each snapshot in the trajectory, 
            the forces, energy, and local atomic configurations for that snapshot's atoms.
            Obtained from ``generate``

    """

    if not isinstance(path, Path):
        path = Path(path)

    suffix = path.suffix

    if str(suffix) == "out":
        traj = read(path, index=index, format='aims-output')
    elif str(suffix) == ".xyz":
        # Get the ASE traj from xyz
        traj = read(path, index=index, format='extxyz')
    else:
        traj = read(path, index=index)

    data = generate(traj, r_cut, forces_label=forces_label,
                    energy_label=energy_label)

    save(path.parent, r_cut, data)

    return data


def load(path, r_cut):
    """ Load data saved with ``save``

    Args:
        path (Path or string): Name and position of file to load data from
        r_cut (float): Cutoff used

    Returns:
        data (dict): Structure containing, for each snapshot in the trajectory, 
            the forces, energy, and local atomic configurations for that snapshot's atoms.
            Obtained from ``generate``

    """

    if not isinstance(path, Path):
        path = Path(path)

    data = np.load('{}/data_cut={:.2f}.npy'.format(path, r_cut),
                   allow_pickle=True)

    return data.item()


def unpack(data):
    """ From a data dictionary, generate elements, configurations, forces, energies and
        global configurations to be used by the GP module.

    Args:
        data (dict): Structure containing, for each snapshot in the trajectory, 
            the forces, energy, and local atomic configurations for that snapshot's atoms.
            Obtained from ``generate``

    Returns:
        elements (list): Atomic numbers of all atomic species present in the dataset
        confs (list of arrays): List of M by 5 numpy arrays, where M is the number of atoms within
            r_cut from the central one. The first 3 components are positions w.r.t
            the central atom in Angstroms, the fourth is the atomic number of the 
            central atom, the fifth the atomic number of each atom.
        forces (array): x,y,z components of the force acting on the central atom in eV/Angstrom
        energies (array): value of the total energy in eV
        global_confs (list of lists of arrays): list containing lists of configurations, grouped together
            so that local atomic environments taken from the same snapshot are in the same group.

    """

    elements = data['elements']
    global_confs = []
    forces = []
    energies = []
    for i in data['data']:
        global_confs.append(i['confs'])
        forces.append(i['forces'])
        energies.append(i['energy'])

    try:
        forces = np.array([item for sublist in forces for item in sublist])
    except:
        logger.warning("No forces in the data file")
    confs = np.array([item for sublist in global_confs for item in sublist])
    try:
        energies = np.array(energies)
    except:
        logger.warning("No energies in the data file")
    global_confs = np.array(global_confs)

    return elements, confs, forces, energies, global_confs


def load_and_unpack(path, r_cut):
    """ Load data saved with ``save`` and unpack it with ``unpak``

    Args:
        path (Path or string): Name and position of file to load data from
        r_cut (float): Cutoff used

    Returns:
        elements (list): Atomic numbers of all atomic species present in the dataset
        confs (list of arrays): List of M by 5 numpy arrays, where M is the number of atoms within
            r_cut from the central one. The first 3 components are positions w.r.t
            the central atom in Angstroms, the fourth is the atomic number of the 
            central atom, the fifth the atomic number of each atom.
        forces (array): x,y,z components of the force acting on the central atom in eV/Angstrom
        energies (array): value of the total energy in eV
        global_confs (list of lists of arrays): list containing lists of configurations, grouped together
            so that local atomic environments taken from the same snapshot are in the same group.

    """
    data = load(path, r_cut)
    elements, confs, forces, energies, global_confs = unpack(data)

    return elements, confs, forces, energies, global_confs
