import os
import logging

import numpy as np
from ase.io import read

from original import better_MFF_database
from original.better_MFF_database import carve_confs

better_MFF_database.USE_ASAP = True
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    r_cut = 100
    n_data = 3000

    directory = 'data/BIP_300/'
    filename = directory + 'movie.xyz'

    # Open file and get number of atoms and steps
    traj = read(filename, index=slice(None), format='extxyz')

    # fixing cell settings
    for atoms in traj:
        atoms.set_cell([r_cut, r_cut, r_cut])

    elements, confs, forces, energies = carve_confs(traj, r_cut, n_data, forces_label='force')

    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save('{}/confs_cut={:.2f}.npy'.format(directory, r_cut), confs)
    np.save('{}/forces_cut={:.2f}.npy'.format(directory, r_cut), forces)
    np.save('{}/energies_cut={:.2f}.npy'.format(directory, r_cut), energies)

    lens = [len(conf) for conf in confs]

    logging.info('\n'.join((
        'Number of atoms in a configuration:',
        '   maximum: {}'.format(np.max(lens)),
        '   minimum: {}'.format(np.min(lens)),
        '   average: {:.4}'.format(np.mean(lens))
    )))
