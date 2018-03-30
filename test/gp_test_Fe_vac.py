import os
from pathlib import Path
import numpy as np
from original import GP_for_MFF
from original import Kernels

if __name__ == '__main__':

    # Parameters
    r_cut = 4.45
    nbodies = 3
    sigma = 1.0
    noise = 0.00001
    ntr = 20
    ntest = 10
    directory = Path('data/Fe_vac/')

    # Get configurations and forces from file
    confs = np.load(str(directory / 'confs_cut={:.2f}.npy'.format(r_cut)))
    forces = np.load(str(directory / 'forces_cut={:.2f}.npy'.format(r_cut)))

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
        NotImplementedError('Kernel order not understood, use 2 for two-body and 3 for three-body')

    gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)
    gp_name = 'gp_ker={}_ntr={}_sig={:.2f}_cut={:.2f}'.format(nbodies, ntr, sigma, r_cut)

    if os.path.isfile(str(directory / gp_name)):
        gp.load(str(directory / gp_name))

    else:
        gp.fit(tr_confs, tr_forces)
        gp.save(str(directory / gp_name))

    # Test the GP performance
    if ntest:
        print('Testing the GP module')

        gp_forces = np.zeros((ntest, 3))
        gp_error = np.zeros((ntest, 3))

        for i in np.arange(ntest):
            gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
            gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]

        print('MAEF on forces: {:.4f} +- {:.4f}'.format(
            np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1))),
            np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))))

# Saved Gaussian process with name: data/Fe_vac/gp_ker=2_ntr=10_sig=1.00_cut=4.45
# Testing the GP module
# MAEF on forces: 0.3387 +- 0.2625

# Saved Gaussian process with name: data/Fe_vac/gp_ker=3_ntr=10_sig=1.00_cut=4.45
# Testing the GP module
# MAEF on forces: 0.2377 +- 0.1257

# Saved Gaussian process with name: data/Fe_vac/gp_ker=3_ntr=20_sig=1.00_cut=4.45
# Testing the GP module
# MAEF on forces: 0.1626 +- 0.0593
#
# gp.predict_energy(tst_confs[0][np.newaxis,...])
# Out[2]: array([[-34011.54]])
# gp.predict_energy(tst_confs[1][np.newaxis,...])
# Out[3]: array([[17301.706]])
# gp.predict_energy(tst_confs[3][np.newaxis,...])
# Out[4]: array([[87888.376]])
# gp.predict_energy(tst_confs[4][np.newaxis,...])
# Out[5]: array([[131479.773]])
