# -*- coding: utf-8 -*-


import json
import warnings
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from mff import gp, interpolation, kernels, utility
from mff.models.base import Model


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class TwoBodySingleSpeciesModel(Model):
    """ 2-body single species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        element (int): The atomic number of the element considered
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (method): The 2-body single species Gaussian Process
        grid (method): The 2-body single species tabulated potential
        grid_start (float): Minimum atomic distance for which the grid is defined (cannot be 0.0)
        grid_num (int): number of points used to create the 2-body grid

    """

    def __init__(self, element, r_cut, sigma, theta, noise, rep_sig=1, **kwargs):
        super().__init__()

        self.element = element
        self.r_cut = r_cut
        self.rep_sig = rep_sig

        kernel = kernels.TwoBodySingleSpeciesKernel(
            theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        2-body single species force-force kernel

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        if self.rep_sig:
            self.rep_sig = utility.find_repulstion_sigma(confs)
            self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
            forces -= self.rep_forces

        self.gp.fit(confs, forces, ncores=ncores)

    def fit_energy(self, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 
        2-body single species energy-energy kernel

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        if self.rep_sig:
            self.rep_sig = utility.find_repulstion_sigma(glob_confs)
            self.rep_energies = utility.get_repulsive_energies(
                glob_confs, self.rep_sig)
            energies -= self.rep_energies

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        2-body single species force-force, energy-force and energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation

        """
        if self.rep_sig:
            self.rep_sig = utility.find_repulstion_sigma(confs)
            self.rep_energies = utility.get_repulsive_energies(
                glob_confs, self.rep_sig)
            energies -= self.rep_energies
            self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
            forces -= self.rep_forces

        self.gp.fit_force_and_energy(
            confs, forces, glob_confs, energies, ncores=ncores)

    def predict(self, confs, return_std=False, ncores=1):
        """ Predict the forces acting on the central atoms of confs using a GP

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            forces (array): array of force vectors predicted by the GP
            forces_errors (array): errors associated to the force predictions,
                returned only if return_std is True

        """

        if self.rep_sig:
            rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
            return self.gp.predict(confs, return_std, ncores=ncores) + rep_forces

        else:
            return self.gp.predict(confs, return_std, ncores=ncores)

    def predict_energy(self, glob_confs, return_std=False, ncores=1):
        """ Predict the global energies of the central atoms of confs using a GP

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            energies (array) : Array containing the total energy of each snapshot
            energies_errors (array): errors associated to the energies predictions,
                returned only if return_std is True

        """
        if self.rep_sig:
            rep_energies = utility.get_repulsive_energies(
                glob_confs, self.rep_sig)
            return self.gp.predict_energy(glob_confs, return_std, ncores=ncores) + rep_energies
        else:
            return self.gp.predict_energy(glob_confs, return_std, ncores=ncores)

    def save_gp(self, filename):
        """ Saves the GP object, now obsolete
        """

        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.save(filename)

    def load_gp(self, filename):
        """ Loads the GP object, now obsolete
        """

        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.load(filename)

    def build_grid(self, start, num, ncores=1):
        """ Build the mapped 2-body potential. 
        Calculates the energy predicted by the GP for two atoms at distances that range from
        start to r_cut, for a total of num points. These energies are stored and a 1D spline
        interpolation is created, which can be used to predict the energy and, through its
        analytic derivative, the force associated to any couple of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        pairwise contributions of every other atom within a cutoff distance r_cut.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 2-body mapped potential
            num (int): number of points to use in the grid of the mapped potential   

        """

        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)

        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element
        confs = list(confs)

        grid_data = self.gp.predict_energy(confs, ncores=ncores, mapping=True)
        if self.rep_sig:
            grid_data += utility.get_repulsive_energies(
                confs, self.rep_sig, mapping=True)
        self.grid = interpolation.Spline1D(dists, grid_data)

    def save(self, path):
        """ Save the model.
        This creates a .json file containing the parameters of the model and the
        paths to the GP objects and the mapped potential, which are saved as 
        separate .gpy and .gpz files, respectively.

        Args:
            path (str): path to the file 

        """

        if not isinstance(path, Path):
            path = Path(path)

        params = {
            'model': self.__class__.__name__,
            'element': self.element,
            'r_cut': self.r_cut,
            'rep_sig': self.rep_sig,
            'fitted': self.gp.fitted,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {
                'r_min': self.grid_start,
                'r_num': self.grid_num,
                'filename': None
            } if self.grid else {}
        }

        gp_filename = "GP_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npy".format(
            p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(path / gp_filename)

        if self.grid:
            grid_filename = 'GRID_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npz'.format(
                p=params)

            params['grid']['filename'] = grid_filename
            self.grid.save(path / grid_filename)

        with open(path / 'MODEL_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.json'.format(p=params), 'w') as fp:
            json.dump(params, fp, indent=4, cls=NpEncoder)

        print("Saved model with name: MODEL_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.json".format(p=params))

    @classmethod
    def from_json(cls, path):
        """ Load the model.
        Loads the model, the associated GP and the mapped potential, if available.

        Args:
            path (str): path to the .json model file 

        Return:
            model (obj): the model object

        """

        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem
        with open(path) as fp:
            params = json.load(fp)

        model = cls(params['element'],
                    params['r_cut'],
                    params['gp']['sigma'],
                    params['gp']['theta'],
                    params['gp']['noise'],
                    params['rep_sig'])

        gp_filename = params['gp']['filename']
        model.gp.load(directory / gp_filename)

        if params['grid']:
            grid_filename = params['grid']['filename']
            model.grid = interpolation.Spline1D.load(directory / grid_filename)

            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

        return model


class TwoBodyManySpeciesModel(Model):
    """ 2-body many species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        elements (list): List containing the atomic numbers in increasing order
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (class): The 2-body two species Gaussian Process
        grid (list): Contains the three 2-body two species tabulated potentials, accounting for
            interactions between two atoms of types 0-0, 0-1, and 1-1.
        grid_start (float): Minimum atomic distance for which the grid is defined (cannot be 0)
        grid_num (int): number of points used to create the 2-body grids

    """

    def __init__(self, elements, r_cut, sigma, theta, noise, rep_sig=1, **kwargs):
        super().__init__()

        self.elements = list(np.sort(elements))
        self.r_cut = r_cut
        self.rep_sig = rep_sig

        kernel = kernels.TwoBodyManySpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = {}, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        2-body single species force-force kernel

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        if self.rep_sig:
            self.rep_sig = utility.find_repulstion_sigma(confs)
            self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
            forces -= self.rep_forces

        self.gp.fit(confs, forces, ncores=ncores)

    def fit_energy(self, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 
        2-body single species energy-energy kernel

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        if self.rep_sig:
            self.rep_sig = utility.find_repulstion_sigma(glob_confs)
            self.rep_energies = utility.get_repulsive_energies(
                glob_confs, self.rep_sig)
            energies -= self.rep_energies

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        2-body single species force-force, energy-force and energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation

        """
        if self.rep_sig:
            self.rep_sig = utility.find_repulstion_sigma(confs)
            self.rep_energies = utility.get_repulsive_energies(
                glob_confs, self.rep_sig)
            energies -= self.rep_energies
            self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
            forces -= self.rep_forces

        self.gp.fit_force_and_energy(
            confs, forces, glob_confs, energies, ncores=ncores)

    def predict(self, confs, return_std=False, ncores=1):
        """ Predict the forces acting on the central atoms of confs using a GP

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            forces (array): array of force vectors predicted by the GP
            forces_errors (array): errors associated to the force predictions,
                returned only if return_std is True

        """

        if self.rep_sig:
            rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
            return self.gp.predict(confs, return_std, ncores=ncores) + rep_forces

        else:
            return self.gp.predict(confs, return_std, ncores=ncores)

    def predict_energy(self, glob_confs, return_std=False, ncores=1):
        """ Predict the global energies of the central atoms of confs using a GP

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            energies (array) : Array containing the total energy of each snapshot
            energies_errors (array): errors associated to the energies predictions,
                returned only if return_std is True

        """
        if self.rep_sig:
            rep_energies = utility.get_repulsive_energies(
                glob_confs, self.rep_sig)
            return self.gp.predict_energy(glob_confs, return_std, ncores=ncores) + rep_energies
        else:
            return self.gp.predict_energy(glob_confs, return_std, ncores=ncores)

    def save_gp(self, filename):
        """ Saves the GP object, now obsolete
        """

        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.save(filename)

    def load_gp(self, filename):
        """ Loads the GP object, now obsolete
        """

        warnings.warn('use save and load function', DeprecationWarning)
        self.gp.load(filename)

    def build_grid(self, start, num, ncores=1):
        """ Build the mapped 2-body potential. 
        Calculates the energy predicted by the GP for two atoms at distances that range from
        start to r_cut, for a total of num points. These energies are stored and a 1D spline
        interpolation is created, which can be used to predict the energy and, through its
        analytic derivative, the force associated to any couple of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        pairwise contributions of every other atom within a cutoff distance r_cut.
        Three distinct potentials are built for interactions between atoms of type 0 and 0, 
        type 0 and 1, and type 1 and 1.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 2-body mapped potential
            num (int): number of points to use in the grid of the mapped potential   

        """
        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)
        # defining the first row heading
        confs = np.zeros((num, 1, 5))
        confs[:, 0, 0] = dists

        perm_list = list(combinations_with_replacement(self.elements, 2))

        for pair in perm_list:  # in this for loop, predicting then save for each individual one
            confs[:, 0, 3], confs[:, 0,
                                  4] = pair[0], pair[1]
            grid_data = self.gp.predict_energy(
                list(confs), ncores=ncores, mapping=True)
            if self.rep_sig:
                grid_data += utility.get_repulsive_energies(
                    confs, self.rep_sig, mapping=True)
            self.grid[pair] = interpolation.Spline1D(dists, grid_data)

    def save(self, path):
        """ Save the model.
        This creates a .json file containing the parameters of the model and the
        paths to the GP objects and the mapped potentials, which are saved as 
        separate .gpy and .gpz files, respectively.

        Args:
            path (str): path to the file 

        """

        if not isinstance(path, Path):
            path = Path(path)

        params = {
            'model': self.__class__.__name__,
            'elements': self.elements,
            'r_cut': self.r_cut,
            'rep_sig': self.rep_sig,
            'fitted': self.gp.fitted,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {
                'r_min': self.grid_start,
                'r_num': self.grid_num,
                'filename': {}
            } if self.grid else {}
        }

        gp_filename = "GP_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npy".format(
            p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(path / gp_filename)

        for k, grid in self.grid.items():
            key = '_'.join(str(element) for element in k)
            grid_filename = "GRID_{}_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npz".format(
                key, p=params)
            params['grid']['filename'][key] = grid_filename
            grid.save(path / grid_filename)

        with open(path / "MODEL_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.json".format(p=params), 'w') as fp:
            json.dump(params, fp, indent=4, cls=NpEncoder)

        print("Saved model with name: MODEL_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.json".format(p=params))

    @classmethod
    def from_json(cls, path):
        """ Load the models.
        Loads the model, the associated GP and the mapped potential, if available.

        Args:
            path (str): path to the .json model file 

        Return:
            model (obj): the model object

        """

        if not isinstance(path, Path):
            path = Path(path)

        directory, prefix = path.parent, path.stem

        with open(path) as fp:
            params = json.load(fp)

        model = cls(params['elements'],
                    params['r_cut'],
                    params['gp']['sigma'],
                    params['gp']['theta'],
                    params['gp']['noise'],
                    params['rep_sig'])

        gp_filename = params['gp']['filename']
        try:
            model.gp.load(directory / gp_filename)
        except:
            warnings.warn("The 2-body GP file is missing")
            pass

        if params['grid']:
            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

            for key, grid_filename in params['grid']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))

                model.grid[k] = interpolation.Spline1D.load(
                    directory / grid_filename)

        return model


if __name__ == '__main__':
    def test_two_body_single_species_model():
        confs = np.array([
            np.hstack([np.random.randn(4, 3), 26 * np.ones((4, 2))]),
            np.hstack([np.random.randn(5, 3), 26 * np.ones((5, 2))])
        ])

        forces = np.random.randn(2, 3)

        element, r_cut, sigma, theta, noise = 26, 2., 3., 4., 5.

        filename = Path('test_model.json')

        m = TwoBodySingleSpeciesModel(element, r_cut, sigma, theta, noise)

        print(m)
        print(m.parameters)

        m.fit(confs, forces)

        print(m)
        print(m.parameters)

        m.build_grid(1., 10)

        print(m)
        print(m.parameters)

        m.save(filename)

        m2 = TwoBodySingleSpeciesModel.from_json(filename)

    def test_two_body_two_species_model():
        elements = [2, 4]
        confs = np.array([
            np.hstack([np.random.randn(4, 3),
                       np.random.choice(elements, size=(4, 2))]),
            np.hstack([np.random.randn(5, 3),
                       np.random.choice(elements, size=(5, 2))])
        ])

        forces = np.random.randn(2, 3)
        r_cut, sigma, theta, noise = 2., 3., 4., 5.

        filename = Path('test_model.json')

        m = TwoBodyManySpeciesModel(elements, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)

        m2 = TwoBodyManySpeciesModel.from_json(filename)

    # test_two_body_single_species_model()

    test_two_body_two_species_model()
