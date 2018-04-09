import numpy as np
from original import create_MFF_grid
from original import GP_for_MFF
from original import Kernels

if __name__ == '__main__':

    # Parameters
    grid_start = 1.5
    r_cut = 4.45
    nbodies = 2
    grid_spacing = 0.01
    ntr = 2
    sigma = 1.0
    processors = 1
    noise = 0.00001
    directory = 'data/Fe_vac'
    elementslist = [26]

    # Load the GP
    if nbodies == 3:
        ker = Kernels.ThreeBody(theta=[sigma, r_cut / 10.0, r_cut])
    elif nbodies == 2:
        ker = Kernels.TwoBody(theta=[sigma, r_cut / 10.0, r_cut])
    else:
        raise NotImplementedError("Kernel order not understood, use 2 for two-body and 3 for three-body")

    gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)
    gp_name = 'gp_ker={}_ntr={}_sig={:.2f}_cut={:.2f}.npy'.format(nbodies, ntr, sigma, r_cut)
    gp.load(directory + '/' + gp_name)

    # Build mapped grids
    # if len(elementslist) == 2:
    #     mapping = create_MFF_grid.Grids_building(d0=grid_start, df=r_cut,
    #                                              nr=int(1 + (r_cut - grid_start) / grid_spacing), gp=gp,
    #                                              element1=elementslist[0], element2=elementslist[1], nnodes=processors)
    # elif len(elementslist) == 1:
    #     mapping = create_MFF_grid.Grids_building(d0=grid_start, df=r_cut,
    #                                              nr=int(1 + (r_cut - grid_start) / grid_spacing), gp=gp,
    #                                              element1=elementslist[0], nnodes=processors)
    # else:
    #     print("There are more than 2 different elements in the configuration, "
    #           "remapping is not implemented for these systems yet.")
    #     quit()

    start, stop, num = grid_start, r_cut, int(1 + (r_cut - grid_start) / grid_spacing)

    mapping = create_MFF_grid.SingleSpecies(gp, start, stop, num, elementslist[0])
    all_grids = mapping.build_grids()

    remap_name = 'MFF_{}b_ntr_{}_sig_{:.2f}_cut_{:.2f}.npy'.format(nbodies, ntr, sigma, r_cut)

    np.save(directory + '/' + remap_name, all_grids)
    print('Saved mapping with name {}'.format(remap_name))
