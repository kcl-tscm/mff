import logging
from abc import ABCMeta

import numpy as np

from asap3 import FullNeighborList

# from ase.neighborlist import NeighborList


USE_ASAP = True
logger = logging.getLogger(__name__)


class MissingData(Exception):
    pass


class Configurations(metaclass=ABCMeta):
    """Configurations can represent a list of configurations"""

    def __init__(self, confs=None):
        super().__init__()
        self.confs = confs if confs else []


class SingleSpecies(metaclass=ABCMeta):
    pass


class MultiSpecies(metaclass=ABCMeta):
    pass


class Forces(metaclass=ABCMeta):
    def __init__(self, forces=None):
        self.forces = forces if forces else []


class Energies(metaclass=ABCMeta):
    def __init__(self, energy=None):
        self.energy = energy if energy else []


class ConfsSingleForces(Configurations, SingleSpecies, Forces):
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


class ConfsTwoForces(Configurations, SingleSpecies, Forces):
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
    from m_ff.sampling import LinspaceSamplingSingle, RandomSamplingSingle

    testfiles = {
        'BIP_300': '../test/data/BIP_300/movie.xyz',
        'C_a': '../test/data/C_a/data_C.xyz',
        'Fe_vac': '../test/data/Fe_vac/movie.xyz',
        'HNi': '../test/data/HNI/h_ase500.xyz'
    }

    filename = testfiles['Fe_vac']
    traj = read(filename, index=slice(0, 4))
    print(traj)

    confs = ConfsSingleForces(r_cut=4.5)
    for atoms, atom_inds in RandomSamplingSingle(traj, n_target=25):
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
