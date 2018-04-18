import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

from original import better_MFF_database
from original.better_MFF_database import carve_confs
from original import Kernels
from original import GP_for_MFF

better_MFF_database.USE_ASAP = False
logging.basicConfig(level=logging.INFO)

# Parameters
directory = 'data/Fe_vac/'
r_cut = 4.45
sigma = 0.4
noise = 0.0001
ntest = 200
# ----------------------------------------
# Construct a configuration database
# ----------------------------------------

if False:

    n_data = 5000

    filename = directory + 'movie.xyz'

    traj = read(filename, index=slice(None), format='extxyz')

    elements, confs, forces, energies = carve_confs(traj, r_cut, n_data,
                                        forces_label= 'force', energy_label = 'energy', USE_ASAP=True)

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

# ----------------------------------------
# Test a simple GP on the built database
# ----------------------------------------

if False:
    # Parameters
    ntr = 5
    ntest = 50

    # Get configurations and forces from file
    confs = np.load(str(directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
    forces = np.load(str(directory + 'forces_cut={:.2f}.npy'.format(r_cut)))
    numconfs = len(forces)
    ind = np.arange(numconfs)
    ind_tot = np.random.choice(ind, size=ntr + ntest, replace=False)

    # Separate into testing and training dataset
    tr_confs, tr_forces = confs[ind[:ntr]], forces[ind[:ntr]]
    tst_confs, tst_forces = confs[ind[ntr:]], forces[ind[ntr:]]

    ker = Kernels.ThreeBodySingleSpecies(theta=[sigma, r_cut / 10.0, r_cut])

    gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)

    print('Training GP')

    gp.fit(tr_confs, tr_forces)

    # Test the GP performance
    print('Testing GP')

    gp_forces = np.zeros((ntest, 3))
    gp_error = np.zeros((ntest, 3))

    for i in np.arange(ntest):
        gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
        gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]

    MAEF = np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1)))
    SMAEF = np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))
    MF = np.mean(np.linalg.norm(tst_forces, axis=1))

    print(gp_forces)
    print('MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF, SMAEF))
    print('Relative MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF / MF, SMAEF / MF))

# ----------------------------------------
# Learning curve
# ----------------------------------------

if True:

	ker = Kernels.TwoBodySingleSpecies(theta=[sigma, r_cut / 10.0, r_cut])
	gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)

	ntrs = [10, 20, 50, 100]

	errors = []

	for ntr in ntrs:

		# Get configurations and forces from file
		confs = np.load(str(directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
		forces = np.load(str(directory + 'forces_cut={:.2f}.npy'.format(r_cut)))

		numconfs = len(forces)
		ind = np.arange(numconfs)
		ind_tot = np.random.choice(ind, size = ntr + ntest, replace=False)

		# Separate into testing and training dataset
		tr_confs, tr_forces = confs[ind_tot[:ntr]], forces[ind_tot[:ntr]]
		tst_confs, tst_forces = confs[ind_tot[ntr:]], forces[ind_tot[ntr:]]

		print('Training GP')

		gp.fit(tr_confs, tr_forces)

		# Test the GP performance
		print('Testing GP')

		gp_forces = np.zeros((ntest, 3))
		gp_error = np.zeros((ntest, 3))

		for i in np.arange(ntest):
			gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
			gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]

		MAEF = np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1)))
		SMAEF = np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))
		MF = np.mean(np.linalg.norm(tst_forces, axis=1))

		print('MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF, SMAEF))
		print('Relative MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF / MF, SMAEF / MF))

		errors.append(MAEF / MF)

	plt.plot(ntrs, errors)
	plt.xscale('log')
	plt.show()

# ----------------------------------------
# Map the GP on an M-FF
# ----------------------------------------
