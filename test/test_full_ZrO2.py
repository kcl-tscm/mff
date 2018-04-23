
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

import sys
sys.path.insert(0, '../')

from original.better_MFF_database import carve_confs
from original import Kernels
from original import GP_for_MFF


logging.basicConfig(level=logging.INFO)

# Parameters
r_cut = 3.

# GP Parameters
sigma = 1.
noise = 0.001


# ----------------------------------------
# Construct a configuration database
# ----------------------------------------

if False:

	n_data = 5000
	directory = 'data/ZrO2/test/'
	filename = directory + 'test_monoclinic.xyz'

	traj = read(filename, index=slice(None), format='extxyz')

	elements, confs, forces, energies = carve_confs(traj, r_cut, n_data, USE_ASAP=True)

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
# Check the configurations created
# ----------------------------------------

if False:
	directory = 'data/ZrO2/train/'
	filename = directory + 'train.xyz'
	confs = np.load(str(directory + 'confs_cut={:.2f}.npy'.format(r_cut)))

	distances = []
	M_mean = 0.
	M_std = 0.
	for c in confs:
		distances += np.linalg.norm(c[:, 0:3], axis = 1).tolist()
		M_mean += c.shape[0]
		M_std += c.shape[0]**2

	M_mean = M_mean/len(confs)
	M_std = M_std/len(confs) - M_mean**2
	print('M is {} += {}'.format(M_mean, M_std))
	plt.hist(distances, 40)
	plt.show()


# ----------------------------------------
# Test GP on the built database
# ----------------------------------------

if False:

	# Directories
	train_directory = 'data/ZrO2/train/'
	test_directory = 'data/ZrO2/test/'

	# Parameters
	ntr = 10
	ntest = 50

	print('N train: {}, N test: {}'.format(ntr, ntest))

	# Get training configurations and forces from file
	tr_confs = np.load(str(train_directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
	tr_forces = np.load(str(train_directory + 'forces_cut={:.2f}.npy'.format(r_cut)))
	numconfs = len(tr_forces)
	ind = np.arange(numconfs)
	ind_tot = np.random.choice(ind, size=ntr, replace=False)
	tr_confs, tr_forces = tr_confs[ind_tot], tr_forces[ind_tot]

	# Get test configurations and forces from file
	tst_confs = np.load(str(test_directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
	tst_forces = np.load(str(test_directory + 'forces_cut={:.2f}.npy'.format(r_cut)))
	numconfs = len(tst_forces)
	ind = np.arange(numconfs)
	ind_tot = np.random.choice(ind, size=ntest, replace=False)
	tst_confs, tst_forces = tst_confs[ind_tot], tst_forces[ind_tot]


	ker = Kernels.ThreeBody(theta=[sigma, r_cut / 8.0, r_cut])

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

	MAEC = np.mean(abs(gp_error))
	MAEF = np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1)))
	SMAEF = np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))
	MF = np.mean(np.linalg.norm(tst_forces, axis=1))
	RMSE = np.sqrt(np.mean((gp_error) ** 2))

	print('RMSE: {:.4}'.format(RMSE))
	print('MAEC: {:.4}'.format(MAEC))
	print('MAEF : {:.4f} +- {:.4f}'.format(MAEF, SMAEF))
	print('Relative MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF / MF, SMAEF / MF))

# ----------------------------------------
# Learning curve
# ----------------------------------------

if True:
	# Directories
	train_directory = 'data/ZrO2/train/'
	test_directory = 'data/ZrO2/test/'

	ker = Kernels.ThreeBody(theta=[sigma, r_cut / 5.0, r_cut])
	gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)

	ntrs = [10, 20, 50, 100, 200, 400]
	ntest = 50
	print('N train: {}, N test: {}'.format(ntrs, ntest))
	errors = []

	for ntr in ntrs:

		# Get training configurations and forces from file
		tr_confs = np.load(str(train_directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
		tr_forces = np.load(str(train_directory + 'forces_cut={:.2f}.npy'.format(r_cut)))
		numconfs = len(tr_forces)
		ind = np.arange(numconfs)
		ind_tot = np.random.choice(ind, size=ntr, replace=False)
		tr_confs, tr_forces = tr_confs[ind_tot], tr_forces[ind_tot]

		# Get test configurations and forces from file
		tst_confs = np.load(str(test_directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
		tst_forces = np.load(str(test_directory + 'forces_cut={:.2f}.npy'.format(r_cut)))
		numconfs = len(tst_forces)
		ind = np.arange(numconfs)
		ind_tot = np.random.choice(ind, size=ntest, replace=False)
		tst_confs, tst_forces = tst_confs[ind_tot], tst_forces[ind_tot]

		print('Training GP')

		gp.fit(tr_confs, tr_forces)

		# Test the GP performance
		print('Testing GP')

		gp_forces = np.zeros((ntest, 3))
		gp_error = np.zeros((ntest, 3))




		for i in np.arange(ntest):

			gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
			gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]


		MAEC = np.mean(abs(gp_error))
		MAEF = np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1)))
		SMAEF = np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))
		RMSE = np.sqrt(np.mean((gp_error)**2))
		
		MF = np.mean(np.linalg.norm(tst_forces, axis=1))

		print('RMSE: {:.4}'.format(RMSE))
		print('MAEC: {:.4}'.format(MAEC))
		print('MAEF: {:.4f} +- {:.4f}'.format(MAEF, SMAEF))
		print('Relative MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF / MF, SMAEF / MF))

		errors.append(MAEF / MF)

	plt.plot(ntrs, errors)
	plt.xscale('log')
	plt.show()

# ----------------------------------------
# Map the GP on an M-FF
# ----------------------------------------
