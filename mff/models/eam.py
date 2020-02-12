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


def get_max_eam(X, rc, r0):
    t_max = 0
    for c in X:
        dist = np.sum(c[:, :3]**2, axis=1)**0.5
        cut_1 = 0.5*(1 + np.cos(np.pi*dist/rc))
        t1 = np.exp(-(dist/r0 - 1))
        t2 = -sum(cut_1*t1)**0.5
        if t2 < t_max:
            t_max = t2
    return t_max

def get_max_eam_energy(X_glob, rc, r0):
    t_max = 0
    for X in X_glob:
        t2 = get_max_eam(X, rc, r0)
        if t2 < t_max:
            t_max = t2
    return t_max

class EamSingleSpeciesModel(Model):
    """ Eam single species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        element (int): The atomic number of the element considered
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (method): The eam single species Gaussian Process
        grid (method): The eam single species tabulated potential
        grid_start (float): Minimum descriptor value for which the grid is defined
        grid_end (float): Maximum descriptor value for which the grid is defined
        grid_num (int): number of points used to create the eam multi grid

    """

    def __init__(self, element, r_cut, sigma, r0, noise, **kwargs):
        super().__init__()

        self.element = element
        self.r_cut = r_cut

        kernel = kernels.EamSingleSpeciesKernel(
            theta=[sigma, r_cut, r0])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_end, self.grid_num = None, None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        eam single species force-force kernel

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
        eam single species energy-energy kernel

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        eam single species force-force, energy-force and energy-energy kernels

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

    def build_grid(self, num, ncores=1):
        """ Build the mapped eam potential. 
        Calculates the energy predicted by the GP for a configuration which eam descriptor
        is evalued between start and end. These energies are stored and a 1D spline
        interpolation is created, which can be used to predict the energy and, through its
        analytic derivative, the force associated to any embedded atom.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            num (int): number of points to use in the grid of the mapped potential   
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """
        
        if 'force' in self.gp.fitted:
            self.grid_start = 3.0 * \
                get_max_eam(self.gp.X_train_, self.r_cut,
                            self.gp.kernel.theta[2])
        else:
           self.grid_start = 3.0 * \
                get_max_eam_energy(self.gp.X_glob_train_, self.r_cut,
                            self.gp.kernel.theta[2])
        self.grid_end = 0
        self.grid_num = num

        dists = list(np.linspace(self.grid_start,
                                 self.grid_end, self.grid_num))

        grid_data = self.gp.predict_energy(dists, ncores=ncores, mapping=True)
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
            'fitted': self.gp.fitted,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'noise': self.gp.noise,
                'r0': self.gp.kernel.theta[2]
            },
            'grid': {
                'r_min': self.grid_start,
                'r_max': self.grid_end,
                'r_num': self.grid_num,
                'filename': {}
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
                    params['gp']['noise'],
                    params['gp']['r0'])

        gp_filename = params['gp']['filename']
        model.gp.load(directory / gp_filename)

        if params['grid']:
            grid_filename = params['grid']['filename']
            model.grid = interpolation.Spline1D.load(directory / grid_filename)

            model.grid_start = params['grid']['r_min']
            model.grid_end = params['grid']['r_max']
            model.grid_num = params['grid']['r_num']

        return model

class EamManySpeciesModel(Model):
    """ Eam many species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        elements (int): The atomic numbers of the element considered
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        r0 (float): radius in the exponent of the eam descriptor
        noise (float): noise value associated with the training output data

    Attributes:
        gp (method): The eam single species Gaussian Process
        grid (method): The eam single species tabulated potential
        grid_start (float): Minimum descriptor value for which the grid is defined
        grid_end (float): Maximum descriptor value for which the grid is defined
        grid_num (int): number of points used to create the eam multi grid

    """

    def __init__(self, elements, r_cut, sigma, r0, noise, **kwargs):
        super().__init__()

        self.elements = list(np.sort(elements))
        self.r_cut = r_cut

        kernel = kernels.EamManySpeciesKernel(
            theta=[sigma, r_cut, r0])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_end, self.grid_num = {}, None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        eam single species force-force kernel

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
        eam single species energy-energy kernel

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        eam single species force-force, energy-force and energy-energy kernels

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

    def build_grid(self, num, ncores=1):
        """ Build the mapped eam potential. 
        Calculates the energy predicted by the GP for a configuration which eam descriptor
        is evalued between start and end. These energies are stored and a 1D spline
        interpolation is created, which can be used to predict the energy and, through its
        analytic derivative, the force associated to any embedded atom.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            num (int): number of points to use in the grid of the mapped potential   
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        if 'force' in self.gp.fitted:
            self.grid_start = 3.0 * \
                get_max_eam(self.gp.X_train_, self.r_cut,
                            self.gp.kernel.theta[2])
        else:
           self.grid_start = 3.0 * \
                get_max_eam_energy(self.gp.X_glob_train_, self.r_cut,
                            self.gp.kernel.theta[2])
                            
        self.grid_end = 0
        self.grid_num = num

        dists = list(np.linspace(self.grid_start,
                                 self.grid_end, self.grid_num))

        for el in self.elements:
            grid_data = self.gp.predict_energy(dists, ncores=ncores, mapping=True, alpha_1_descr=el)
            self.grid[(el)] = interpolation.Spline1D(dists, grid_data)

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
            'elements': self.elements,
            'r_cut': self.r_cut,
            'fitted': self.gp.fitted,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'noise': self.gp.noise,
                'r0': self.gp.kernel.theta[2]
            },
            'grid': {
                'r_min': self.grid_start,
                'r_max': self.grid_end,
                'r_num': self.grid_num,
                'filename': {}
            } if self.grid else {}
        }

        gp_filename = "GP_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npy".format(
            p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(path / gp_filename)

        for k, grid in self.grid.items():
            key = str(k)
            grid_filename = "GRID_{}_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.npz".format(
                key, p=params)
            params['grid']['filename'][key] = grid_filename
            grid.save(path / grid_filename)

        with open(path / "MODEL_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.json".format(p=params), 'w') as fp:
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
                    params['gp']['noise'],
                    params['gp']['r0'])

        gp_filename = params['gp']['filename']
        model.gp.load(directory / gp_filename)

        if params['grid']:
            model.grid_start = params['grid']['r_min']
            model.grid_end = params['grid']['r_max']
            model.grid_num = params['grid']['r_num']

            for key, grid_filename in params['grid']['filename'].items():
                k = tuple(key)

                model.grid[k] = interpolation.Spline1D.load(
                    directory / grid_filename)

        return model

