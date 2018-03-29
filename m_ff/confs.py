from ase.io import extxyz
import numpy as np

from ase import Atom, Atoms

from ase.geometry import find_mic

from ase.io import iread

from scipy import spatial
from asap3 import FullNeighborList
from ase.neighborlist import NeighborList



class Confs(object):
    def __init__(self, atomic_numbers, positions,
                 forces=None,
                 potential_energy=None):
        """

        Args:
            atomic_numbers (ndarray):
            positions (ndarray):
            forces (ndarray):
            potential_energy (float):
        """

        self.atomic_number = atomic_numbers
        self.positions = positions
        self.forces = forces
        self.potential_energy = potential_energy

    @classmethod
    def from_atoms(cls, atoms, r_cut, index=None):
        atomic_numbers = atoms.get_array('numbers', copy=False)
        # positions = atoms.get_array('positions', copy=False)

        confs = []
        atomic_numbers = []
        forces = []
        for atomic_number, conf, force in cls.get_confs_cKDTree(atoms, r_cut):
            atomic_numbers.append(atomic_number)
            confs.append(conf)



        return cls(atomic_numbers, confs)

    # @staticmethod
    # def get_confs(atoms, r_cut):
    #     cutoffs = np.ones(len(atoms)) * r_cut / 2
    #     nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False, bothways=True)
    #     nl.build(atoms)
    #
    #     cell = atoms.get_cell()
    #
    #     for atom in atoms:
    #         indices, offsets = nl.get_neighbors(atom.index)
    #         confs = atoms.positions[indices, :] + np.dot(offsets, cell) - atom.position
    #
    #         yield confs

    @staticmethod
    def get_confs_asap(atoms, r_cut):
        # https://wiki.fysik.dtu.dk/asap/Neighbor%20lists

        atomic_numbers = atoms.get_array('numbers', copy=False)
        nl = FullNeighborList(r_cut, atoms=atoms)

        for atom in atoms:
            inds, confs, ds = nl.get_neighbors(atom.index)



            yield atomic_numbers[inds], confs, force

    @staticmethod
    def get_confs_cKDTree(atoms, r_cut):
        # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.spatial.cKDTree.html
        atomic_numbers = atoms.get_array('numbers', copy=False)
        positions = atoms.get_positions(wrap=True)
        box = np.diag(atoms.cell)

        tree = spatial.cKDTree(positions, boxsize=box)
        nl = tree.query_ball_point(positions, r_cut)

        for index, indices in enumerate(nl):
            indices.remove(index)

            confs = positions[indices] - positions[index]

            # fixing periodic boundary
            confs = np.where(abs(confs) < abs(confs - box), confs, confs - box)
            confs = np.where(abs(confs) < abs(confs + box), confs, confs + box)

            yield atomic_numbers[indices], confs


if __name__ == '__main__':
    atoms = Atoms('H2',
                  positions=[[0, 0, 0],
                             [0, 0, 0.7]])

    confs = Confs.from_atoms(atoms)


class Configuration(object):
    """Class for representing a configuration
    """

    def __init__(self, atomic_number, positions, configurations=None):
        """

        Args:
            number: atomic number
        """
        self.atomic_number = atomic_number
        self.positions = np.array(positions, float)

        self.configurtions = configurations
        self._distances = None

    @property
    def distances(self):
        if not self._distances:
            self._distances = 0

        return self._distances


class Configurations(object):
    """Configurations can represent a list of configurations"""
    pass


class ConfsForce(Configurations):
    """Hello

    Args:
        r_cut:
    """

    def __init__(self, r_cut):
        """test

        Args:
            r_cut:
        """
        self.r_cut = r_cut

        self.confs = None
        self.forces = None
        self.energies = None

    def __iter__(self):
        return self

    def __next__(self):
        return None

    @classmethod
    def from_npy(cls):
        pass

    @classmethod
    def from_ase(cls, atoms, r_cut):
        return cls(r_cut)

    @classmethod
    def from_file(cls, filename, r_cut):
        return cls(r_cut)

#
# def carve_confs(filename, r_cut):
#     ### Open file and get number of atoms and steps ###
#     f = open(filename, 'r')
#     N = int(f.readline())
#
#     num_lines = 1 + sum(1 for line in f)
#     f.close()
#     steps = num_lines / (N + 2)
#     n_data = float(min(5000, N * steps))
#     print('Database will have %i entries' % (n_data))
#
#     ### Read the number and types of elements ###
#     atoms = extxyz.read_extxyz(filename, index=0)
#     atoms = next(atoms)
#     elementslist = list(set(atoms.get_atomic_numbers()))
#     elplace = []
#
#     for i in np.arange(len(elementslist)):
#         elplace.append(np.where(atoms.get_atomic_numbers() == elementslist[i]))
#     elplace = np.array(elplace)
#
#     confs = []
#     forces = []
#     energies = []
#
#     if len(elementslist) == 2:
#         print("There are 2 elements in the XYZ file")
#         ### Choose the number of entries each element will have, proportional to the square root of the ratio of occurrences of each element ###
#         ratio = np.sqrt(len(elplace[0, 0]) / float(len(elplace[1, 0])))
#         nc_el1 = int(n_data * ratio / (1.0 + ratio))
#         nc_el2 = n_data - nc_el1
#
#         cutoffs = np.ones(N) * r_cut / 2.
#         nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False, bothways=True)
#
#         ### Build conf and forces database centered on element 1 ###
#         for i in np.arange(nc_el1):
#             print('step %i' % (i))
#             j = int(i * float(steps) / nc_el1)
#             atoms = extxyz.read_extxyz(filename, index=j)
#             atoms = next(atoms)
#             nl.build(atoms)
#             cell = atoms.get_cell()
#             ind_atom = int(
#                 i % len(elplace[0, 0]))  # Select the atom number rotationg between the total atoms of el 1 present #
#
#             d = np.array([atoms.arrays['positions'][elplace[0, 0]][ind_atom]])
#             errvalue = 0
#             try:
#                 force = atoms.get_array('force')[elplace[0, 0]][ind_atom]
#             except KeyError:
#                 print('Forces in the xyz file are not present, or are not called force')
#                 force = None
#                 errvalue += 1
#             try:
#                 energy = atoms.get_array('energy')[elplace[0, 0]][ind_atom]
#             except KeyError:
#                 print('Energies in the xyz file are not present, or are not called energy')
#                 energy = None
#                 errvalue += 1
#
#             if errvalue == 2:
#                 print('Cannot find energy or force values in the xyz file, shutting down now')
#                 quit()
#
#             indices, offsets = nl.get_neighbors(ind_atom)
#             offsets = np.dot(offsets, cell)
#             conf = np.zeros((len(indices), 5))
#
#             for k, (a2, offset) in enumerate(zip(indices, offsets)):
#                 d = atoms.positions[a2] + offset - atoms.positions[elplace[0, 0]][ind_atom]
#                 conf[k, :3] = d
#                 conf[k, 4] = atoms.get_atomic_numbers()[
#                     a2]  # Set the last digit of confs to be the element of the atom in the conf
#
#             conf[:, 3] = atoms.get_atomic_numbers()[elplace[0, 0]][
#                 ind_atom]  # Set the fourth digit of confs to be the element of the central atom
#             confs.append(conf)
#             forces.append(force)
#             energies.append(energy)
#
#         ### Build conf and forces database centered on element 2, exact same procedure ###
#         for i in np.arange(nc_el2):
#             print('step %i' % (i + nc_el1))
#             j = int(i * float(steps) / nc_el2)
#             atoms = extxyz.read_extxyz(filename, index=j)
#             atoms = next(atoms)
#             nl.build(atoms)
#             cell = atoms.get_cell()
#             ind_atom = int(i % len(elplace[1, 0]))
#
#             d = np.array([atoms.arrays['positions'][elplace[1, 0]][ind_atom]])
#             errvalue = 0
#             try:
#                 force = atoms.get_array('force')[elplace[1, 0]][ind_atom]
#             except KeyError:
#                 print('Forces in the xyz file are not present, or are not called force')
#                 force = None
#                 errvalue += 1
#             try:
#                 energy = atoms.get_array('energy')[elplace[1, 0]][ind_atom]
#             except KeyError:
#                 print('Energies in the xyz file are not present, or are not called energy')
#                 energy = None
#                 errvalue += 1
#
#             if errvalue == 2:
#                 print('Cannot find energy or force values in the xyz file, shutting down now')
#                 quit()
#
#             indices, offsets = nl.get_neighbors(ind_atom)
#             offsets = np.dot(offsets, cell)
#             conf = np.zeros((len(indices), 5))
#
#             for k, (a2, offset) in enumerate(zip(indices, offsets)):
#                 d = atoms.positions[a2] + offset - atoms.positions[elplace[1, 0]][ind_atom]
#                 conf[k, :3] = d
#                 conf[k, 4] = atoms.get_atomic_numbers()[a2]
#
#             conf[:, 3] = atoms.get_atomic_numbers()[elplace[1, 0]][ind_atom]
#             confs.append(conf)
#             forces.append(force)
#             energies.append(energy)
#
#     else:
#         print("There is 1 element in the XYZ file")
#
#         ### Choose the number of entries each element will have, proportional to the square root of the ratio of occurrences of each element ###
#         nc_el1 = n_data
#
#         cutoffs = np.ones(N) * r_cut / 2.
#         nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False, bothways=True)
#
#         ### Build conf and forces database centered on element 1 ###
#         for i in np.arange(nc_el1):
#             print('step %i' % (i))
#             j = int(i * float(steps) / nc_el1)
#             atoms = extxyz.read_extxyz(filename, index=j)
#             atoms = next(atoms)
#             nl.build(atoms)
#             cell = atoms.get_cell()
#             ind_atom = int(i % N)  # Select the atom number rotationg between the total atoms of el 1 present #
#
#             d = np.array([atoms.arrays['positions'][ind_atom]])
#             errvalue = 0
#             try:
#                 force = atoms.get_array('force')[ind_atom]
#             except KeyError:
#                 print('Forces in the xyz file are not present, or are not called force')
#                 force = None
#                 errvalue += 1
#             try:
#                 energy = atoms.get_array('energy')[ind_atom]
#             except KeyError:
#                 print('Energies in the xyz file are not present, or are not called energy')
#                 energy = None
#                 errvalue += 1
#
#             if errvalue == 2:
#                 print('Cannot find energy or force values in the xyz file, shutting down now')
#                 quit()
#
#             indices, offsets = nl.get_neighbors(ind_atom)
#             offsets = np.dot(offsets, cell)
#             conf = np.zeros((len(indices), 5))
#
#             for k, (a2, offset) in enumerate(zip(indices, offsets)):
#                 d = atoms.positions[a2] + offset - atoms.positions[ind_atom]
#                 conf[k, :3] = d
#                 conf[k, 4] = atoms.get_atomic_numbers()[
#                     a2]  # Set the last digit of confs to be the element of the atom in the conf
#
#             conf[:, 3] = atoms.get_atomic_numbers()[
#                 ind_atom]  # Set the fourth digit of confs to be the element of the central atom
#             confs.append(conf)
#             forces.append(force)
#             energies.append(energy)
#
#     forces = np.array(forces)
#     energies = np.array(energies)
#
#     np.save("confs_cut=%.2f.npy" % (r_cut), confs)
#     np.save("forces_cut=%.2f.npy" % (r_cut), forces)
#     np.save("energies__cut=%.2f.npy" % (r_cut), energies)
#     lens = []
#     for i in np.arange(len(confs)):
#         lens.append(len(confs[i]))
#
#     print(max(lens))
#     print(min(lens))
#     print(np.mean(lens))
#
#     return elementslist
#
#
# # carve_confs('movie.xyz', 4.31)
#
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
#         # confs2 = list(get_confs_asap(atoms, r_cut)) # 2 sec 16x
#         # confs3 = list(get_confs_cKDTree(atoms, r_cut)) # 0.6 sec 50x
#
#         for conf, force in zip(get_confs_asap(atoms, r_cut), forces):
#             confs_list.append(conf)
#             force_list.append(force)
