import json
from collections import defaultdict

from ase import Atoms
from ase.io import read

from mff.configurations import ConfsTwoBodySingleForces
from mff.sampling import SamplingLinspaceSingle
from mff.gp import GaussianProcess
from mff import kernels
from mff.calculators import Calculator

if __name__ == '__main__':

    filename = 'test/data/Fe_vac/movie.xyz'
    traj = read(filename, index=slice(0, 10))

    # Storing parameters
    jsonfile = 'data.json'
    p = defaultdict(dict)

    # keeping track of the parameters manually
    p['r_cut'] = 4.5
    p['configurations']['n_target'] = 25

    # ------------------------------
    # Carving
    # ------------------------------

    confs = ConfsTwoBodySingleForces(r_cut=p['r_cut'])

    for atoms, atom_inds in SamplingLinspaceSingle(traj, n_target=p['configurations']['n_target']):
        confs.append(atoms, atom_inds)

    confs.save(filename_conf='aaa', filename_force='aaa')

    # ------------------------------
    # GP
    # ------------------------------

    kernel = kernels.TwoBodySingleSpeciesKernel()

    # gp = GaussianProcess(kernel=kernel, theta=..., noise=...)
    gp = GaussianProcess()

    gp.fit(confs)
    gp.save(filename='.npy')

    # ------------------------------
    # Mapped
    # ------------------------------

    grid = MappedTwoBodySingleSpecies(gp)
    grid.save(filename='.npy')

    with open(jsonfile, 'w') as file:
        json.dumps(file, p)

    # ------------------------------
    # Calculator
    # ------------------------------

    atoms = Atoms()

    filename = 'sadfsd.json'
    calc = Calculator.from_json(filename)

    atoms.set_calculator(calc)

    for step in range(100):
        pass

# model = Model(nbody=2, elements=[11], r_cut=4.5)
# for atoms, atom_inds in LinspaceSamplingSingle(traj, n_target=25):
#     model.confs.append(atoms, atom_inds)
#
# model.savejson()

# calc = Calculator(model)
