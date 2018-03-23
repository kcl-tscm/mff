from ase import Atoms
from m_ff.calculators import Calculator
from m_ff.configurations import ConfsSingleForces
from m_ff.sampling import LinspaceSamplingSingle

if __name__ == '__main__':

    traj = [Atoms(), Atoms()]

    parameter = {
        "configurations:": {},
        "gaussianprocess": {},
        "remappedpotential": {}
    }

    # keeping track of the parameters manually
    parameter['r_cut'] = 4.5
    parameter['configurations']['n_target'] = 25

    # Carving
    confs = ConfsTwoBodySingleForces(r_cut=4.5)
    for atoms, atom_inds in LinspaceSamplingSingle(traj, n_target=25):
        confs.append(atoms, atom_inds)

    confs.save(numpyfilename='sadfasdf')
    #

    # GP
    kernel = TwoBodySingle(...)
    gp = GP(kernel= kernel, theta=..., noise=...)

    gp.fit(confs)
    gp.save(filename='.npy')

    # Remapping

    grid = MapTwoBodySingle(gp)
    grid.save(filename='.npy')

    with open(jsonfile) as file:
        json.dumps(file, parameter)

    # Calculator

    atoms = Atoms()

    filename = 'sadfsd.json'
    calc = Calculator(model)
    calc = Calculator.from_json(filename)

    atoms.set_calculator(calc)

    for step in range(100):
        pass

# model = Model(nbody=2, elements=[11], r_cut=4.5)
# for atoms, atom_inds in LinspaceSamplingSingle(traj, n_target=25):
#     model.confs.append(atoms, atom_inds)
#
# model.savejson()
