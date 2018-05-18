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
sigma = 0.6
noise = 0.0001
ntest = 20

combine_2b_3b = False
# ----------------------------------------
# Construct a configuration database
# ----------------------------------------

if False:

    n_data = 5000

    filename = directory + 'movie.xyz'

    traj = read(filename, index=slice(None), format='extxyz')

    elements, confs, forces, energies = carve_confs(traj, r_cut, n_data,
                                                    forces_label='force', energy_label='energy', USE_ASAP=True)

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
    # ntr = 100
    ntr = 10
    ntest = 10

    # Get configurations and forces from file
    confs = np.load(str(directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
    forces = np.load(str(directory + 'forces_cut={:.2f}.npy'.format(r_cut)))
    numconfs = len(forces)
    ind = np.arange(numconfs)
    ind_tot = np.random.choice(ind, size=ntr + ntest, replace=False)

    # Separate into random testing and training dataset
    # tr_confs, tr_forces = confs[ind[:ntr]], forces[ind[:ntr]]
    # tst_confs, tst_forces = confs[ind[ntr:]], forces[ind[ntr:]]

    # Use fixed training and testing dataset
    tr_confs, tr_forces = confs[:ntr], forces[:ntr]
    tst_confs, tst_forces = confs[-ntest - 1:-1], forces[-ntest - 1:-1]

    two_body_train_forces = np.zeros((ntr, 3))

    if combine_2b_3b:
        # First train with a 2 body
        ker_2 = Kernels.TwoBodySingleSpecies(theta=[sigma / 2.0, r_cut / 10.0, r_cut])
        gp_2 = GP_for_MFF.GaussianProcess(kernel=ker_2, noise=noise, optimizer=None)
        print('Training 2B GP')
        gp_2.fit(tr_confs, tr_forces)

        gp_2_name = 'gp_ker=2_ntr={}_sig={:.2f}_cut={:.2f}'.format(ntr, sigma, r_cut)
        gp_2.save(directory + gp_2_name)

        # Calculate the predictions of the 2body on the training set
        for i in np.arange(ntr):
            two_body_train_forces[i] = gp_2.predict(np.reshape(tr_confs[i], (1, len(tr_confs[i]), 5)))


    # Then train with a 2 body on the difference between tr_force and tr_force obtained with the two body
    ker_3 = Kernels.ThreeBodySingleSpecies(theta=[sigma, r_cut / 10.0, r_cut])
    gp_3 = GP_for_MFF.GaussianProcess(kernel=ker_3, noise=noise, optimizer=None)
    print('Training 3B GP')
    gp_3.fit(tr_confs, tr_forces - two_body_train_forces)

    gp_3_name = 'gp_ker=3_ntr={}_sig={:.2f}_cut={:.2f}'.format(ntr, sigma, r_cut)
    gp_3.save(directory + gp_3_name)

    # Test the GP performance
    print('Testing GP')

    gp_3b_forces = np.zeros((ntest, 3))
    gp_error = np.zeros((ntest, 3))
    gp_2b_error = np.zeros((ntest, 3))
    gp_2b_forces = np.zeros((ntest, 3))

    for i in np.arange(ntest):
        if combine_2b_3b:
            gp_2b_forces[i, :] = gp_2.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
            gp_2b_error[i, :] = gp_2b_forces[i, :] - tst_forces[i, :]
        gp_3b_forces[i, :] = gp_3.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
        gp_error[i, :] = gp_2b_forces[i, :] + gp_3b_forces[i, :] - tst_forces[i, :]

    MAEF = np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1)))
    SMAEF = np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))
    MAEF_2B = np.mean(np.sqrt(np.sum(np.square(gp_2b_error), axis=1)))
    SMAE_2B = np.std(np.sqrt(np.sum(np.square(gp_2b_error), axis=1)))

    MF = np.mean(np.linalg.norm(tst_forces, axis=1))

    print('2 body MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF_2B, SMAE_2B))
    print('MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF, SMAEF))
    print('Relative MAEF on forces: {:.4f} +- {:.4f}'.format(MAEF / MF, SMAEF / MF))



# ----------------------------------------
# Testing
# ----------------------------------------





# ----------------------------------------
# Learning curve
# ----------------------------------------

if True:

    ker = Kernels.TwoBodySingleSpecies(theta=[sigma/ 2.0, r_cut / 10.0, r_cut])
    # ker = kernels.ThreeBodySingleSpecies(theta=[sigma, r_cut / 10.0, r_cut])
    gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)

    ntrs = [10,20, 40, 100, 140, 200]

    errors = []
    for ntr in ntrs:

        # Get configurations and forces from file
        confs = np.load(str(directory + 'confs_cut={:.2f}.npy'.format(r_cut)))
        forces = np.load(str(directory + 'forces_cut={:.2f}.npy'.format(r_cut)))

        numconfs = len(forces)
        ind = np.arange(numconfs)
        ind_tot = np.random.choice(ind, size=ntr + ntest, replace=False)

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
    # plt.xscale('log')
    plt.show()

# ----------------------------------------
# Map the GP on an M-FF
# ----------------------------------------
