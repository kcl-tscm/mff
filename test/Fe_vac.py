# import numpy as np
#
# if __name__ == '__main__':
#     directory = 'data/Fe_vac/'
#     filename = 'MFF_3b_ntr_3_sig_1.00_cut_4.45.npy'
#
#     # directory = 'data/C_a/'
#     # filename = 'MFF_2b_ntr_10_sig_1.00_cut_3.70.npy'
#
#     rs, element1, element2, grid_1_1, grid_1_1_1 = np.load(directory + filename)

import os
import numpy as np
from original import GP_for_MFF
from original import Kernels
from original import create_MFF_grid




if __name__ == '__main__':

    # Parameters
    # r_cut = 4.45
    # nbodies = 2
    # sigma = 1.0
    # noise = 0.00001
    # ntr = 10
    # ntest = 10
    # directory = 'data/Fe_vac'
    #
    # # Parameters
    # grid_start = 1.5
    # grid_spacing = 0.1
    # processors = 1
    # elementslist = [26]
    #
    # # Get configurations and forces from file
    # confs = np.load('{:s}/confs_cut={:.2f}.npy'.format(directory, r_cut))
    # forces = np.load('{:s}/forces_cut={:.2f}.npy'.format(directory, r_cut))
    # numconfs = len(forces)
    # ind = np.arange(numconfs)
    # ind_tot = np.random.choice(ind, size=ntr + ntest, replace=False)
    #
    # # Separate into testing and training dataset
    # tr_confs, tr_forces = confs[ind[:ntr]], forces[ind[:ntr]]
    # tst_confs, tst_forces = confs[ind[ntr:]], forces[ind[ntr:]]
    #
    # # Load, or train, the GP
    # if nbodies == 3:
    #     ker = Kernels.ThreeBody(theta=[sigma, r_cut / 10.0, r_cut])
    # elif nbodies == 2:
    #     ker = Kernels.TwoBody(theta=[sigma, r_cut / 10.0, r_cut])
    # else:
    #     print("Kernel order not understood, use 2 for two-body and 3 for three-body")
    #     quit()
    #
    # gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)
    # gp_name = 'gp_ker=%s_ntr=%i_sig=%.2f_cut=%.2f' % (nbodies, ntr, sigma, r_cut)
    #
    #
    # gp.fit(tr_confs, tr_forces)
    # gp.save(directory + '/' + gp_name)
    #
    # # Test the GP performance
    # if ntest > 0:
    #     print("Testing the GP module")
    #     gp_forces = np.zeros((ntest, 3))
    #     gp_error = np.zeros((ntest, 3))
    #     for i in np.arange(ntest):
    #         gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
    #         gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]
    #     print("MAEF on forces: %.4f +- %.4f" % (
    #         np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1))),
    #         np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))))
    #
    #
    #
    # # Build mapped grids
    # if len(elementslist) == 2:
    #     mapping = create_MFF_grid.Grids_building(d0=grid_start, df=r_cut,
    #                                              nr=int(1 + (r_cut - grid_start) / grid_spacing), gp=gp,
    #                                              element1=elementslist[0], element2=elementslist[1], nnodes=processors)
    # elif len(elementslist) == 1:
    #     mapping = create_MFF_grid.Grids_building(d0=grid_start, df=r_cut,
    #                                              nr=int(1 + (r_cut - grid_start) / grid_spacing), gp=gp,
    #                                              element1=elementslist[0], nnodes=processors)
    # else:
    #     print(
    #         "There are more than 2 different elements in the configuration, remapping is not implemented for these systems yet.")
    #     quit()
    #
    # all_grids = mapping.build_grids()
    # remap_name = ("MFF_%ib_ntr_%i_sig_%.2f_cut_%.2f.npy" % (nbodies, ntr, sigma, r_cut))
    # np.save(directory+'/'+remap_name, all_grids)
    # print('Saved mapping with name %s' % remap_name)