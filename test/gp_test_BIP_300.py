import os
import numpy as np
from original import GP_for_MFF
from original import Kernels

# import sys
#
# sys.path.append("../original/")

if __name__ == '__main__':

    # Parameters
    r_cut = 100.0
    nbodies = 2
    sigma = 1.0
    noise = 0.00001
    ntr = 11
    ntest = 100
    directory = 'data/BIP_300'

    # Get configurations and forces from file
    confs = np.load('{:s}/confs_cut={:.2f}.npy'.format(directory, r_cut))
    forces = np.load('{:s}/forces_cut={:.2f}.npy'.format(directory, r_cut))
    numconfs = len(forces)
    ind = np.arange(numconfs)
    ind_tot = np.random.choice(ind, size=ntr + ntest, replace=False)

    # Separate into testing and training dataset
    tr_confs, tr_forces = confs[ind[:ntr]], forces[ind[:ntr]]
    tst_confs, tst_forces = confs[ind[ntr:]], forces[ind[ntr:]]

    # Load, or train, the GP
    if nbodies == 3:
        ker = Kernels.ThreeBody(theta=[sigma, r_cut / 10.0, r_cut])
    elif nbodies == 2:
        ker = Kernels.TwoBody(theta=[sigma, r_cut / 10.0, r_cut])
    else:
        print("Kernel order not understood, use 2 for two-body and 3 for three-body")
        quit()

    gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)
    gp_name = 'gp_ker=%s_ntr=%i_sig=%.2f_cut=%.2f' % (nbodies, ntr, sigma, r_cut)

    if os.path.isfile(directory + '/' + gp_name):
        gp.load(directory + '/' + gp_name)

    else:
        gp.fit(tr_confs, tr_forces)
        gp.save(directory + '/' + gp_name)

    # Test the GP performance
    if (ntest > 0):
        print("Testing the GP module")
        gp_forces = np.zeros((ntest, 3))
        gp_error = np.zeros((ntest, 3))
        for i in np.arange(ntest):
            gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
            gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]
        print("MAEF on forces: %.4f +- %.4f" % (
            np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1))),
            np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))))
