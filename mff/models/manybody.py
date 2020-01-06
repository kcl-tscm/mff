# -*- coding: utf-8 -*-


import json
import warnings
from pathlib import Path

import numpy as np

from mff import gp, interpolation, kernels
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


class ManyBodySingleSpeciesModel(Model):
    """ many-body single species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        element (int): The atomic number of the element considered
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (method): The many-body single species Gaussian Process
        grid (method): The many-body single species tabulated potential
        grid_start (float): Minimum atomic distance for which the grid is defined (cannot be 0.0)
        grid_num (int): number of points used to create the many-body grid

    """

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()

        self.element = element
        self.r_cut = r_cut

        kernel = kernels.ManyBodySingleSpeciesKernel(
            theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        many-body single species force-force kernel

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit(confs, forces, ncores=ncores)

    def fit_energy(self, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 
        many-body single species energy-energy kernel

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        many-body single species force-force, energy-force and energy-energy kernels

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

        directory, prefix = path.parent, path.stem

        params = {
            'model': self.__class__.__name__,
            'element': self.element,
            'r_cut': self.r_cut,
            'fitted': self.gp.fitted,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {}
        }

        gp_filename = "GP_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npy".format(
            p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(path / gp_filename)

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
                    params['gp']['noise'])

        gp_filename = params['gp']['filename']
        try:
            model.gp.load(directory / gp_filename)
        except:
            warnings.warn("The many-body GP file is missing")
            pass

        return model


class ManyBodyManySpeciesModel(Model):
    """ many-body many species model class
    Class managing the Gaussian process, there is no mapping method for this kernel.

    Args:
        elements (list): List containing the atomic numbers in increasing order
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (class): The many-body two species Gaussian Process
        grid (list): None
        grid_start (float): None
        grid_num (int): None

    """

    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()

        self.elements = elements
        self.r_cut = r_cut

        kernel = kernels.ManyBodyManySpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = {}, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a two 
        body two species force-force kernel

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation

        """

        self.gp.fit(confs, forces, ncores=ncores)

    def fit_energy(self, glob_confs, energy, ncores=1):
        """ Fit the GP to a set of training energies using a two 
        body two species energy-energy kernel

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation

        """

        self.gp.fit_energy(glob_confs, energy, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energy, ncores=1):
        """ Fit the GP to a set of training forces and energies using two 
        body two species force-force, energy-force and energy-energy kernels

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

        self.gp.fit_force_and_energy(
            confs, forces, glob_confs, energy, ncores=ncores)

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

        directory, prefix = path.parent, path.stem

        params = {
            'model': self.__class__.__name__,
            'elements': self.elements,
            'r_cut': self.r_cut,
            'fitted': self.gp.fitted,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {}
        }

        gp_filename = "GP_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npy".format(
            p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(path / gp_filename)

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
        model = cls(params['elements'],
                    params['r_cut'],
                    params['gp']['sigma'],
                    params['gp']['theta'],
                    params['gp']['noise'])

        gp_filename = params['gp']['filename']
        try:
            model.gp.load(directory / gp_filename)
        except:
            warnings.warn("The many-body GP file is missing")
            pass

        return model
