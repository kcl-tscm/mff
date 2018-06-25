import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from asap3 import FullNeighborList

logger = logging.getLogger(__name__)


class MissingData(Exception):
    pass


def carve_from_snapshot(atoms, atoms_ind, r_cut, forces_label=None, energy_label=None):
    # See if there are forces and energies, get them for the chosen atoms
    if (atoms.get_cell() == np.zeros((3,3))).all():
        atoms.set_cell(100.0*np.identity(3))
        logging.warning('No cell values found, setting to a 100 x 100 x 100 cube')

    forces = atoms.arrays.get(forces_label) if forces_label else atoms.get_forces()
    if (energy_label and energy_label != 'energy'):
        energy = atoms.arrays.get(energy_label) 
    else:
        try:
            energy = atoms.get_potential_energy() 
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

    # Build local configurations for every indexed atom
    nl = FullNeighborList(r_cut, atoms=atoms)
    confs = []
    for i in atoms_ind:
        indices, positions, distances = nl.get_neighbors(i)

        atomic_numbers_i = np.ones((len(indices), 1)) * atoms.get_atomic_numbers()[i]
        atomic_numbers_j = atoms.get_atomic_numbers()[indices].reshape(-1, 1)

        confs.append(np.hstack([positions, atomic_numbers_i, atomic_numbers_j]))
    return confs, forces, energy

def carve_confs(atoms, r_cut, n_data, forces_label=None, energy_label=None, smart_sampling = False):
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
        if smart_sampling:
            for k in np.arange(len(elements)):
                count_el_atoms = sum(atom_number_list[j] == elements[k])
                element_ind_count[k] += count_el_atoms
                temp_ind = np.array([x for x in (indices[k] - element_ind_count_prev[k]) if (0 <= x < count_el_atoms)],
                                    dtype=np.int)

                this_ind.append((np.where(atom_number_list[j] == elements[k]))[0][temp_ind])
                element_ind_count_prev[k] += count_el_atoms

            this_ind = np.concatenate(this_ind).ravel()
        else:
            this_ind = np.asarray(np.arange(len(atoms[j])))

        # Call the carve_from_snapshot function on the chosen atoms
        if this_ind.size > 0:
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
    from m_ff.sampling import SamplingLinspaceSingle, SamplingRandomSingle

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

#
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
