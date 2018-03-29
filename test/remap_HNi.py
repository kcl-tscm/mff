import numpy as np
from original import create_MFF_grid
from original import GP_for_MFF
from original import Kernels

# Parameters
grid_start = 1.5
cutoff = 4.5
nbodies = 3
grid_spacing = 0.1
ntr = 10
sigma = 1.0
processors = 1
noise = 0.00001
directory = 'data/HNi/'
elementslist = [1, 28]

# Load the GP
if nbodies == 3:
    ker = Kernels.ThreeBody(theta=[sigma, cutoff / 10.0, cutoff])
elif nbodies == 2:
    ker = Kernels.TwoBody(theta=[sigma, cutoff / 10.0, cutoff])
else:
    raise NotImplementedError('Kernel order not understood, use 2 for two-body and 3 for three-body')

gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=noise, optimizer=None)
gp_name = 'gp_ker={}_ntr={}_sig={:.2f}_cut={:.2f}.npy'.format(nbodies, ntr, sigma, cutoff)
gp.load(directory + gp_name)

# # Build mapped grids
# if len(elementslist) == 2:
#     mapping = create_MFF_grid.Grids_building(d0=grid_start, df=cutoff,
#                                              nr=int(1 + (cutoff - grid_start) / grid_spacing), gp=gp,
#                                              element1=elementslist[0], element2=elementslist[1], nnodes=processors)
# elif len(elementslist) == 1:
#     mapping = create_MFF_grid.Grids_building(d0=grid_start, df=cutoff,
#                                              nr=int(1 + (cutoff - grid_start) / grid_spacing), gp=gp,
#                                              element1=elementslist[0], nnodes=processors)
# else:
#     print(
#         "There are more than 2 different elements in the configuration, remapping is not implemented for these systems yet.")
#     quit()

start, stop, num = grid_start, cutoff, int(1 + (cutoff - grid_start) / grid_spacing)

mapping = create_MFF_grid.TwoSpecies(gp, start, stop, num, elementslist[0], elementslist[1])
all_grids = mapping.build_grids()

remap_name = 'MFF_{}b_ntr_{}_sig_{:.2f}_cut_{:.2f}_new.npy'.format(nbodies, ntr, sigma, cutoff)

np.save(directory + remap_name, all_grids)
print('Saved mapping with name {}'.format(remap_name))
