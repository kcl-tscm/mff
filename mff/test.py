import os
from os import listdir
from os.path import isfile, join
import logging
import numpy as np
from ase import Atoms
import mff
from mff import models, calculators, utility
from mff import configurations as cfg


def get_potential(confs):
    pot = 0
    for conf in confs:
        el1 = conf[:, 3]
        el2 = conf[:, 4]
        dist = np.sum(conf[:, :3]**2, axis=1)**0.5
        pot += np.sum(el1**0.5*el2**0.5*pot_profile(dist))
    return pot


def pot_profile(dist):
    return ((dist-1)**2 - 0.5)*np.exp(-dist)


def force_profile(dist):
    a = (dist-1)**2 - 0.5
    da = 2*(dist-1)
    b = np.exp(-dist)
    db = -np.exp(-dist)

    return a*db+b*da


def get_potentials(many_confs):
    pots = np.zeros(len(many_confs))
    for i, confs in enumerate(many_confs):
        pots[i] = get_potential(confs)
    return pots


def get_force(conf):
    el1 = conf[:, 3]
    el2 = conf[:, 4]
    dist = np.sum(conf[:, :3]**2, axis=1)**0.5
    vers = conf[:, :3]/dist[:, None]
    force = np.sum(vers * (el1[:, None]**0.5*el2[:, None]
                           ** 0.5*(force_profile(dist[:, None]))), axis=0)
    return force


def get_forces(many_confs):
    forces = np.zeros((len(many_confs), 3))
    for i, confs in enumerate(many_confs):
        forces[i] = get_force(confs)
    return forces


def generate_confs(n, elements, r_cut):
    phi = np.random.uniform(0, 2*np.pi, size=n*2)
    costheta = np.random.uniform(-1, 1, size=n*2)
    u = np.random.uniform(0, 1, size=n*2)

    theta = np.arccos(costheta)
    r = r_cut * u**(1/3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    xyz = np.vstack((x, y, z)).T

    glob_confs = []
    loc_confs = []

    for i in range(n):
        conf1 = np.zeros((2, 5))
        conf2 = np.zeros((2, 5))
        conf3 = np.zeros((2, 5))

        conf1[0, :3] = xyz[2*i]
        conf1[1, :3] = xyz[2*i+1]
        conf2[0, :3] = -xyz[2*i]
        conf2[1, :3] = xyz[2*i+1] - xyz[2*i]
        conf3[0, :3] = xyz[2*i] - xyz[2*i+1]
        conf3[1, :3] = -xyz[2*i+1]

        if len(elements) == 1:
            conf1[:, 3] = elements
            conf1[:, 4] = elements
            conf2[:, 3] = elements
            conf2[:, 4] = elements
            conf3[:, 3] = elements
            conf3[:, 4] = elements

        elif len(elements) >= 2:
            a, b, c = np.random.choice(elements), np.random.choice(
                elements), np.random.choice(elements)
            conf1[:, 3] = a
            conf1[0, 4] = b
            conf1[1, 4] = c
            conf2[:, 3] = b
            conf2[0, 4] = a
            conf2[1, 4] = c
            conf3[:, 3] = c
            conf3[0, 4] = b
            conf3[1, 4] = a

        this_conf = np.array([conf1, conf2, conf3])
        glob_confs.append(this_conf)
        loc_confs.append(conf1)
        loc_confs.append(conf2)
        loc_confs.append(conf3)

    loc_confs = np.array(loc_confs)
    glob_confs = np.array(glob_confs)

    return (glob_confs, loc_confs)


def fit_test(m, loc_confs, forces, glob_confs, energies, ntr, ntest, elements, fit_type, r_cut, ncores = 1):
    if fit_type == 'force':
        m.fit(loc_confs[:ntr], forces[:ntr], ncores=ncores)
    elif fit_type == 'energy':
        m.fit_energy(glob_confs[:ntr], energies[:ntr], ncores=ncores)
    elif fit_type == 'force_and_energy':
        m.fit_force_and_energy(
            loc_confs[:ntr], forces[:ntr], glob_confs[:ntr], energies[:ntr], ncores=ncores)
    pred_forces = m.predict(loc_confs[-ntest:], ncores=ncores)
    pred_energies = m.predict_energy(glob_confs[-ntest:], ncores=ncores)
#     print("MAEF: %.4f eV/A " %(np.mean(np.sum(forces[-ntest:] - pred_forces, axis = 1)**2)**0.5))
#     print("MAEE: %.4f eV" %( np.mean(abs(energies[-ntest:] - pred_energies))))
    mtype = str(type(m)).split('.')[-1].split("'")[0]

    if mtype == "TwoBodySingleSpeciesModel" or mtype == "ThreeBodySingleSpeciesModel" or mtype == "TwoBodyManySpeciesModel" or mtype == "ThreeBodyManySpeciesModel":
        m.build_grid(0.0, 5, ncores=2)
        if mtype == "TwoBodySingleSpeciesModel":
            calc = calculators.TwoBodySingleSpecies(r_cut*2,  m.grid)
        elif mtype == "ThreeBodySingleSpeciesModel":
            calc = calculators.ThreeBodySingleSpecies(r_cut*2,  m.grid)
        elif mtype == "TwoBodyManySpeciesModel":
            calc = calculators.TwoBodyManySpecies(r_cut*2, elements, m.grid)
        elif mtype == "ThreeBodyManySpeciesModel":
            calc = calculators.ThreeBodyManySpecies(r_cut*2, elements, m.grid)

    elif mtype == "CombinedSingleSpeciesModel" or mtype == "CombinedManySpeciesModel":
        m.build_grid(0.0, 5, 5, ncores=2)
        if mtype == "CombinedSingleSpeciesModel":
            calc = calculators.CombinedSingleSpecies(
                r_cut*2,  m.grid_2b, m.grid_3b)
        elif mtype == "CombinedManySpeciesModel":
            calc = calculators.CombinedManySpecies(
                r_cut*2, elements,  m.grid_2b, m.grid_3b)

    elif mtype == "EamSingleSpeciesModel" or mtype == "EamManySpeciesModel":
        m.build_grid(5, ncores=2)
        if mtype == "EamSingleSpeciesModel":
            calc = calculators.EamSingleSpecies(
                r_cut*2,  m.grid, m.gp.kernel.theta[2], m.gp.kernel.theta[3])
        elif mtype == "EamManySpeciesModel":
            calc = calculators.EamManySpecies(
                r_cut*2, elements,  m.grid, m.gp.kernel.theta[2], m.gp.kernel.theta[3])

    elif mtype == "TwoThreeEamSingleSpeciesModel" or mtype == "TwoThreeEamManySpeciesModel":
        m.build_grid(0, 5, 5, 5, ncores=2)
        if mtype == "TwoThreeEamSingleSpeciesModel":
            calc = calculators.TwoThreeEamSingleSpecies(
                r_cut*2,  m.grid_2b, m.grid_3b, m.grid_eam, m.gp_eam.kernel.theta[2], m.gp_eam.kernel.theta[3])
        elif mtype == "TwoThreeEamManySpeciesModel":
            calc = calculators.TwoThreeEamManySpecies(
                r_cut*2, elements,  m.grid_2b, m.grid_3b, m.grid_eam, m.gp_eam.kernel.theta[2], m.gp_eam.kernel.theta[3])

    map_forces = np.zeros((len(pred_forces), 3))
    map_energies = np.zeros_like(pred_energies)

    for i in np.arange(ntest):
        coords = np.vstack(([0, 0, 0], glob_confs[-ntest:][i][0, 0:3, 0:3]))
        atoms = Atoms(positions=coords + 20)
        atoms.set_atomic_numbers([glob_confs[-ntest:][i][0, 0, 3],
                                  glob_confs[-ntest:][i][0, 0, 4], glob_confs[-ntest:][i][0, 1, 4]])
        atoms.set_cell([100, 100, 100])
        atoms.set_calculator(calc)
        map_energies[i] = atoms.get_potential_energy()

    for i in np.arange(ntest):
        coords = np.vstack(([0, 0, 0], loc_confs[-ntest:][i][0:3, 0:3]))
        atoms = Atoms(positions=coords + 20)
        atoms.set_atomic_numbers(
            [loc_confs[-ntest:][i][0, 3], loc_confs[-ntest:][i][0, 4], loc_confs[-ntest:][i][1, 4]])
        atoms.set_cell([100, 100, 100])
        atoms.set_calculator(calc)
        map_forces[i] = atoms.get_forces()[0, :]

    error_f = np.sum((pred_forces - map_forces)**2, axis=1)**0.5
    error_e = pred_energies - map_energies

#     print("Force Error: %.4f eV/A  Energy Error: %.4f eV " %(np.mean(error_f), np.mean(error_e)))
    m.save('MODELS/')


class Tests():
    def __init__(self, elements, noise, sigma, r_cut, theta, ntr_f,
                 ntr_e, ntest, alpha, r0, ncores):

        self.elements = elements
        self.noise = noise
        self.sigma = sigma
        self.r_cut = r_cut
        self.theta = theta
        self.ntr_f = ntr_f
        self.ntr_e = ntr_e
        self.ntest = ntest
        self.alpha = alpha
        self.r0 = r0
        self.ncores = ncores

        self.glob_confs, self.loc_confs = generate_confs(self.ntr_f+self.ntr_e+self.ntest,
                                                         self.elements, self.r_cut)
        self.forces = get_forces(self.loc_confs)
        self.energies = get_potentials(self.glob_confs)

    def test_2_body_single(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.TwoBodySingleSpeciesModel(
                element=self.elements, noise=self.noise, sigma=self.sigma, r_cut=self.r_cut*2,
                theta=self.theta, rep_sig=0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in 2-body Single %s fit" % (fit_type))

    def test_3_body_single(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.ThreeBodySingleSpeciesModel(
                element=self.elements, noise=self.noise, sigma=self.sigma,
                r_cut=self.r_cut*2, theta=self.theta)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in 3-body Single %s fit" % (fit_type))

    def test_combined_body_single(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.CombinedSingleSpeciesModel(
                element=self.elements, noise=self.noise, sigma_2b=self.sigma, sigma_3b=self.sigma, r_cut=self.r_cut*2, theta_2b=self.theta, theta_3b=self.theta, rep_sig=0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in combined Single %s fit" % (fit_type))

    def test_eam_single(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.EamSingleSpeciesModel(
                element=self.elements, noise=self.noise, sigma=self.sigma, r_cut=self.r_cut*2, alpha=self.alpha, r0=self.r0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in Eam Single %s fit" % (fit_type))

    def test_23eam_single(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.TwoThreeEamSingleSpeciesModel(
                self.elements, self.r_cut*2, self.sigma, self.sigma, self.sigma, self.theta, self.theta, self.alpha, self.r0, self.noise, 0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in 23 Eam Single %s fit" % (fit_type))

    def test_2_body_many(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.TwoBodyManySpeciesModel(
                elements=self.elements, noise=self.noise, sigma=self.sigma, r_cut=self.r_cut*2, theta=self.theta, rep_sig=0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in 2-body Many %s fit" % (fit_type))

    def test_3_body_many(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.ThreeBodyManySpeciesModel(
                elements=self.elements, noise=self.noise, sigma=self.sigma, r_cut=self.r_cut*2, theta=self.theta)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in 3-body Many %s fit" % (fit_type))

    def test_combined_body_many(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.CombinedManySpeciesModel(elements=self.elements, noise=self.noise, sigma_2b=self.sigma,
                                                sigma_3b=self.sigma, r_cut=self.r_cut*2, theta_2b=self.theta, theta_3b=self.theta, rep_sig=0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in combined Many %s fit" % (fit_type))

    def test_eam_many(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.EamManySpeciesModel(
                elements=self.elements, noise=self.noise, sigma=self.sigma, r_cut=self.r_cut*2, alpha=self.alpha, r0=self.r0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in eam Many %s fit" % (fit_type))

    def test_23eam_many(self):
        for fit_type in ("force", "energy", "force_and_energy"):
            m = models.TwoThreeEamManySpeciesModel(
                self.elements, self.r_cut*2, self.sigma, self.sigma, self.sigma, self.theta, self.theta, self.alpha, self.r0, self.noise, 0)
            try:
                fit_test(m, self.loc_confs, self.forces, self.glob_confs,
                         self.energies, self.ntr_f, self.ntest, self.elements, fit_type, self.r_cut, self.ncores)
            except:
                print("ERROR in 23 eam Many %s fit" % (fit_type))

    def test_load(self):
        onlyfiles = [f for f in listdir("MODELS") if isfile(join("MODELS", f))]
        for file in onlyfiles:
            if file.endswith(".json"):
                try:
                    m2 = utility.load_model("MODELS/" + file)
                except:
                    print("ERROR: %s not loaded" % (file))


if __name__ == '__main__':
    # GP Parameters
    sigma = 1.0 # Angstrom - typical value 0.2-0.6
    noise = .001  # Number   - Typical values 0.01 - 0.0001
    theta = 0.1 # Cutoff decay lengthscale in Angstrom - Typical value r_cut/5 - r_cut/10
    r_cut = 3.0
    ntr_f = 10
    ntr_e = 10
    ntest = 10
    elements = [1]
    ncores = 2
    alpha = 1
    r0 = 10

    test = Tests(elements, noise, sigma, r_cut, theta, ntr_f,
                    ntr_e, ntest, alpha, r0, ncores)

    test.test_2_body_single()
    test.test_3_body_single()
    test.test_combined_body_single()
    test.test_eam_single()
    test.test_23eam_single()

    test.test_2_body_many()
    test.test_3_body_many()
    test.test_combined_body_many()
    test.test_eam_many()
    test.test_23eam_many()  

    test.test_load()