# -*- coding: utf-8 -*-

import json
import sys
import warnings
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from mff import gp, interpolation, kernels
from mff.models.base import Model

sys.setrecursionlimit(100000)


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


class ThreeBodySingleSpeciesModel(Model):
    """ 3-body single species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        element (int): The atomic number of the element considered
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (method): The 3-body single species Gaussian Process
        grid (method): The 3-body single species tabulated potential
        grid_start (float): Minimum atomic distance for which the grid is defined (cannot be 0.0)
        grid_num (int): number of points per side used to create the 3-body grid. This is a 
            3-dimensional grid, therefore the total number of grid points will be grid_num^3.
    """

    def __init__(self, element, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()
        self.element = element
        self.r_cut = r_cut

        kernel = kernels.ThreeBodySingleSpeciesKernel(
            theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        3-body single species force-force kernel function

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
        3-body single species energy-energy kernel function

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        3-body single species force-force, energy-force and energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
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

    def predict_energy(self, confs, return_std=False, ncores=1):
        """ Predict the global energies of the central atoms of confs using a GP 

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            energies (array): array of force vectors predicted by the GP
            energies_errors (array): errors associated to the energies predictions,
                returned only if return_std is True
        """

        return self.gp.predict_energy(confs, return_std, ncores=ncores)

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
        """ Build the mapped 3-body potential. 
        Calculates the energy predicted by the GP for three atoms at all possible combination 
        of num distances ranging from start to r_cut. The energy is calculated only for ``valid``
        triplets of atoms, i.e. sets of three distances which form a triangle (this is checked via 
        the triangle inequality). The grid building exploits all the permutation invariances to
        reduce the number of energy calculations needed to fill the grid.
        The computed energies are stored in a 3D cube of values, and a 3D spline interpolation is 
        created, which can be used to predict the energy and, through its analytic derivative, 
        the force associated to any triplet of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        triplet contributions of every valid triplet of atoms of which one is always the central one.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 3-body mapped potential
            num (int): number of points to use to generate the list of distances used to
                generate the triplets of atoms for the mapped potential
            ncores (int): number of CPUs to use to calculate the energy predictions
        """

        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))
        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        # Permutations of elements
        confs[:, :, 3] = self.element  # Central element is always element 1
        # Element on the x axis is always element 2
        confs[:, 0, 4] = self.element
        # Element on the xy plane is always element 3
        confs[:, 1, 4] = self.element
        confs = np.nan_to_num(confs)  # Avoid nans to ruin everything
        confs = list(confs)
        grid_data = np.zeros((num, num, num))

        grid_data[inds] = self.gp.predict_energy(
            confs, ncores=ncores, mapping=True).flatten()

        for ind_i in range(num):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_data[ind_i, ind_k,
                              ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_i,
                              ind_k] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_k,
                              ind_i] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_i,
                              ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_j,
                              ind_i] = grid_data[ind_i, ind_j, ind_k]

        self.grid = interpolation.Spline3D(dists, dists, dists, grid_data)

    @staticmethod
    def generate_triplets(dists):
        """ Generate a list of all valid triplets using perutational invariance.
        Calculates the energy predicted by the GP for three atoms at all possible combination 
        of num distances ranging from start to r_cut. The energy is calculated only for ``valid``
        triplets of atoms, i.e. sets of three distances which form a triangle (this is checked via 
        the triangle inequality). The grid building exploits all the permutation invariances to
        reduce the number of energy calculations needed to fill the grid.
        The computed energies are stored in a 3D cube of values, and a 3D spline interpolation is 
        created, which can be used to predict the energy and, through its analytic derivative, 
        the force associated to any triplet of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        triplet contributions of every valid triplet of atoms of which one is always the central one.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            dists (array): array of floats containing all of the distances which can be used to 
                build triplets of atoms. This array is created by calling np.linspace(start, r_cut, num)

        Returns:
            inds (array): array of booleans indicating which triplets (three distance values) need to be
                evaluated to fill the 3D grid of energy values.
            r_ij_x (array): array containing the x coordinate of the second atom j w.r.t. the central atom i
            r_ki_x (array): array containing the x coordinate of the third atom k w.r.t. the central atom i
            r_ki_y (array): array containing the y coordinate of the third atom k w.r.t. the central atom i

        """

        d_ij, d_jk, d_ki = np.meshgrid(
            dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(
            d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Utilizing permutation invariance
        inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 +
                  d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y

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
            'elements': self.element,
            'r_cut': self.r_cut,
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
        model = cls(params['elements'],
                    params['r_cut'],
                    params['gp']['sigma'],
                    params['gp']['theta'],
                    params['gp']['noise'])

        gp_filename = params['gp']['filename']
        try:
            model.gp.load(directory / gp_filename)
        except:
            warnings.warn("The 3-body GP file is missing")
            pass

        if params['grid']:
            grid_filename = params['grid']['filename']
            model.grid = interpolation.Spline3D.load(directory / grid_filename)

            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

        return model


class ThreeBodyManySpeciesModel(Model):
    """ 3-body many species model class
    Class managing the Gaussian process and its mapped counterpart

    Args:
        elements (list): List containing the atomic numbers in increasing order
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma (foat): Lengthscale parameter of the Gaussian process
        theta (float): decay ratio of the cutoff function in the Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp (class): The 3-body two species Gaussian Process
        grid (list): Contains the three 3-body two species tabulated potentials, accounting for
            interactions between three atoms of types 0-0-0, 0-0-1, 0-1-1, and 1-1-1.
        grid_start (float): Minimum atomic distance for which the grid is defined (cannot be 0)
        grid_num (int): number of points per side used to create the 3-body grids. These are 
            3-dimensional grids, therefore the total number of grid points will be grid_num^3.  
    """

    def __init__(self, elements, r_cut, sigma, theta, noise, **kwargs):
        super().__init__()
        self.elements = list(np.sort(elements))
        self.r_cut = r_cut

        kernel = kernels.ThreeBodyManySpeciesKernel(
            theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = {}, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 
        3-body two species force-force kernel function

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
        3-body two species energy-energy kernel function

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_energy(glob_confs, energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training forces and energies using 
        3-body two species force-force, energy-force and energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_force_and_energy(
            confs, forces, glob_confs, energies, ncores=ncores)

    def update_force(self, confs, forces, ncores=1):
        """ Update a fitted GP with a set of forces and using 
        3-body twp species force-force kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation

        """

        self.gp.fit_update(confs, forces, ncores=ncores)

    def update_energy(self, glob_confs, energies, ncores=1):
        """ Update a fitted GP with a set of energies and using 
        3-body two species energy-energy kernels

        Args:
            glob_confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation

        """

        self.gp.fit_update_energy(glob_confs, energies, ncores=ncores)

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
        """ Predict the local energies of the central atoms of confs using a GP 

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            energies (array): array of force vectors predicted by the GP
            energies_errors (array): errors associated to the energies predictions,
                returned only if return_std is True
        """

        return self.gp.predict_energy(glob_confs, return_std, ncores=ncores)

    def save_gp(self, filename):
        """ Saves the GP object, now obsolete
        """

        self.gp.save(filename)

    def load_gp(self, filename):
        """ Loads the GP object, now obsolete
        """

        self.gp.load(filename)

    def build_grid(self, start, num, ncores=1):
        """Function used to create the four different 3-body energy grids for 
        atoms of elements 0-0-0, 0-0-1, 0-1-1, and 1-1-1. The function calls the
        ``build_grid_3b`` function for each of those combinations of elements.

        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 3-body mapped potential
            num (int): number of points to use to generate the list of distances used to
                generate the triplets of atoms for the mapped potential
            ncores (int): number of CPUs to use to calculate the energy predictions
        """

        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)
        perm_list = list(combinations_with_replacement(self.elements, 3))

        for trip in perm_list:

            self.grid[trip] = self.build_grid_3b(
                dists,  trip[0],  trip[1],  trip[2], ncores)

    def build_grid_3b(self, dists, element_i, element_j, element_k, ncores):
        """ Build a mapped 3-body potential. 
        Calculates the energy predicted by the GP for three atoms of elements element_i, element_j, element_k, 
        at all possible combinations of num distances ranging from start to r_cut. 
        The energy is calculated only for ``valid`` triplets of atoms, i.e. sets of three distances 
        which form a triangle (this is checked via the triangle inequality), found by calling the 
        ``generate_triplets_with_permutation_invariance`` function.
        The computed energies are stored in a 3D cube of values, and a 3D spline interpolation is 
        created, which can be used to predict the energy and, through its analytic derivative, 
        the force associated to any triplet of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        triplet contributions of every valid triplet of atoms of which one is always the central one.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            dists (array): array of floats containing all of the distances which can be used to 
                build triplets of atoms. This array is created by calling np.linspace(start, r_cut, num)
            element_i (int): atomic number of the central atom i in a triplet
            element_j (int): atomic number of the second atom j in a triplet
            element_k (int): atomic number of the third atom k in a triplet
            ncores (int): number of CPUs to use when computing the triplet local energies

        Returns:
            spline3D (obj): a 3D spline object that can be used to predict the energy and the force associated
                to the central atom of a triplet.
        """

        num = len(dists)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets_all(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))

        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        confs[:, :, 3] = element_i  # Central element is always element 1
        confs[:, 0, 4] = element_j  # Element on the x axis is always element 2
        # Element on the xy plane is always element 3
        confs[:, 1, 4] = element_k
        confs = np.nan_to_num(confs)  # Avoid nans to ruin everything
        confs = list(confs)

        grid_3b = np.zeros((num, num, num))

        grid_3b[inds] = self.gp.predict_energy(
            confs, ncores=ncores, mapping=True).flatten()

        return interpolation.Spline3D(dists, dists, dists, grid_3b)

    @staticmethod
    def generate_triplets_with_permutation_invariance(dists):
        """ Generate a list of all valid triplets using perutational invariance.
        Calculates the energy predicted by the GP for three atoms at all possible combination 
        of num distances ranging from start to r_cut. The energy is calculated only for ``valid``
        triplets of atoms, i.e. sets of three distances which form a triangle (this is checked via 
        the triangle inequality). The grid building exploits all the permutation invariances to
        reduce the number of energy calculations needed to fill the grid.
        The computed energies are stored in a 3D cube of values, and a 3D spline interpolation is 
        created, which can be used to predict the energy and, through its analytic derivative, 
        the force associated to any triplet of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        triplet contributions of every valid triplet of atoms of which one is always the central one.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            dists (array): array of floats containing all of the distances which can be used to 
                build triplets of atoms. This array is created by calling np.linspace(start, r_cut, num)

        Returns:
            inds (array): array of booleans indicating which triplets (three distance values) need to be
                evaluated to fill the 3D grid of energy values.
            r_ij_x (array): array containing the x coordinate of the second atom j w.r.t. the central atom i
            r_ki_x (array): array containing the x coordinate of the third atom k w.r.t. the central atom i
            r_ki_y (array): array containing the y coordinate of the third atom k w.r.t. the central atom i
        """

        d_ij, d_jk, d_ki = np.meshgrid(
            dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(
            d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Utilizing permutation invariance
        inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 +
                  d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y

    @staticmethod
    def generate_triplets_all(dists):
        """ Generate a list of all valid triplets
        Calculates the energy predicted by the GP for three atoms at all possible combination 
        of num distances ranging from start to r_cut. The energy is calculated only for ``valid``
        triplets of atoms, i.e. sets of three distances which form a triangle (this is checked via 
        the triangle inequality).
        The computed energies are stored in a 3D cube of values, and a 3D spline interpolation is 
        created, which can be used to predict the energy and, through its analytic derivative, 
        the force associated to any triplet of atoms.
        The total force or local energy can then be calculated for any atom by summing the 
        triplet contributions of every valid triplet of atoms of which one is always the central one.
        The prediction is done by the ``calculator`` module which is built to work within 
        the ase python package.

        Args:
            dists (array): array of floats containing all of the distances which can be used to 
                build triplets of atoms. This array is created by calling np.linspace(start, r_cut, num)

        Returns:
            inds (array): array of booleans indicating which triplets (three distance values) need to be
                evaluated to fill the 3D grid of energy values.
            r_ij_x (array): array containing the x coordinate of the second atom j w.r.t. the central atom i
            r_ki_x (array): array containing the x coordinate of the third atom k w.r.t. the central atom i
            r_ki_y (array): array containing the y coordinate of the third atom k w.r.t. the central atom i
        """

        d_ij, d_jk, d_ki = np.meshgrid(
            dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(
            d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 +
                  d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y

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

        print('Saved model with name:', str(
            path / "MODEL_ker_{p[gp][kernel]}_ntr_{p[gp][n_train]}.json".format(p=params)))

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
            warnings.warn("The 3-body GP file is missing")
            pass

        if params['grid']:

            for key, grid_filename in params['grid']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid[k] = interpolation.Spline3D.load(
                    directory / grid_filename)

            model.grid_start = params['grid']['r_min']
            model.grid_num = params['grid']['r_num']

        return model


if __name__ == '__main__':
    def test_three_body_single_species_model():
        confs = np.array([
            np.hstack([np.random.randn(4, 3), 26 * np.ones((4, 2))]),
            np.hstack([np.random.randn(5, 3), 26 * np.ones((5, 2))])
        ])

        forces = np.random.randn(2, 3)

        element, r_cut, sigma, theta, noise = 26, 2., 3., 4., 5.

        filename = Path() / 'test_model'

        m = ThreeBodySingleSpeciesModel(element, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)

    def test_three_body_two_species_model():
        elements = [2, 4]
        confs = np.array([
            np.hstack([np.random.randn(4, 3),
                       np.random.choice(elements, size=(4, 2))]),
            np.hstack([np.random.randn(5, 3),
                       np.random.choice(elements, size=(5, 2))])
        ])

        forces = np.random.randn(2, 3)
        r_cut, sigma, theta, noise = 2., 3., 4., 5.

        filename = Path() / 'test_model'

        m = ThreeBodyManySpeciesModel(elements, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)

    test_three_body_single_species_model()
    # test_three_body_two_species_model()
