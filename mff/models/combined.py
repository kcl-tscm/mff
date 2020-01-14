# -*- coding: utf-8 -*-


import json
import logging
import warnings
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from mff import gp, interpolation, kernels, utility, models

from .base import Model

logger = logging.getLogger(__name__)


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


class CombinedSingleSpeciesModel(Model):
    """ 2- and 3-body single species model class
    Class managing the Gaussian processes and their mapped counterparts

    Args:
        element (int): The atomic number of the element considered
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma_2b (foat): Lengthscale parameter of the 2-body Gaussian process
        sigma_3b (foat): Lengthscale parameter of the 2-body Gaussian process
        theta_2b (float): decay ratio of the cutoff function in the 2-body Gaussian Process
        theta_3b (float): decay ratio of the cutoff function in the 3-body Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp_2b (method): The 2-body single species Gaussian Process
        gp_3b (method): The 3-body single species Gaussian Process
        grid_2b (method): The 2-body single species tabulated potential
        grid_3b (method): The 3-body single species tabulated potential
        grid_start (float): Minimum atomic distance for which the grids are defined (cannot be 0.0)
        grid_num (int): number of points per side used to create the 2- and 3-body grid. The 3-body
            grid is 3-dimensional, therefore its total number of grid points will be grid_num^3    
    """

    def __init__(self, element, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, rep_sig=1, **kwargs):
        super().__init__()
        self.element = element
        self.r_cut = r_cut
        self.rep_sig = rep_sig

        kernel_2b = kernels.TwoBodySingleSpeciesKernel(
            theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(
            kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodySingleSpeciesKernel(
            theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(
            kernel=kernel_3b, noise=noise, **kwargs)

        self.grid_2b, self.grid_3b, self.grid_start, self.grid_num = None, None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 2- and
        3-body single species force-force kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training forces and the 2-body predictions of force on the 
        training configurations

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """

        hypotetical_model_name = ("models/MODEL_ker_TwoBodySingleSpecies_ntr_%i.json" %(len(forces)))
        try:
            model_2b = models.TwoBodySingleSpeciesModel.from_json(hypotetical_model_name)
            self.rep_sig = model_2b.rep_sig
            self.gp_2b = model_2b.gp
            if self.rep_sig:
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces
            print("Loaded 2-body model to bootstart training")

        except:
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(confs)
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces

            self.gp_2b.fit(confs, forces, ncores=ncores)

        two_body_forces = self.gp_2b.predict(confs, ncores=ncores)

        self.gp_3b.fit(confs, forces - two_body_forces, ncores=ncores)

    def fit_energy(self, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body single species energy-energy kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training energies and the 2-body predictions of energies on the 
        training configurations.

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """


        hypotetical_model_name = "models/MODEL_ker_TwoBodySingleSpecies_ntr_%i.json" %(len(energies))
        try:
            model_2b = models.TwoBodySingleSpeciesModel.from_json(hypotetical_model_name)
            self.rep_sig = model_2b.rep_sig
            self.gp_2b = model_2b.gp
            if self.rep_sig:
                self.rep_energies = utility.get_repulsive_energies(
                        glob_confs, self.rep_sig)
                energies -= self.rep_energies
            print("Loaded 2-body model to bootstart training")

        except:
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(glob_confs)
                self.rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                energies -= self.rep_energies

            self.gp_2b.fit_energy(glob_confs, energies, ncores=ncores)

        two_body_energies = self.gp_2b.predict_energy(
            glob_confs, ncores=ncores)

        self.gp_3b.fit_energy(glob_confs, energies -
                              two_body_energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body single species force-force, energy-energy, and energy-forces kernel 
        functions. The 2-body Gaussian process is first fitted, then the 3-body GP 
        is fitted to the difference between the training energies (and forces) and 
        the 2-body predictions of energies (and forces) on the training configurations.

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

        hypotetical_model_name = "models/MODEL_ker_TwoBodySingleSpecies_ntr_%i.json" %(len(energies)+len(forces))
        try:
            model_2b = models.TwoBodySingleSpeciesModel.from_json(hypotetical_model_name)
            self.rep_sig = model_2b.rep_sig
            self.gp_2b = model_2b.gp
            if self.rep_sig:
                self.rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                energies -= self.rep_energies
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces

            print("Loaded 2-body model to bootstart training")

        except:
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(confs)
                self.rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                energies -= self.rep_energies
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces

            self.gp_2b.fit_force_and_energy(
                confs, forces, glob_confs, energies, ncores=ncores)

        two_body_forces = self.gp_2b.predict(confs, ncores=ncores)
        two_body_energies = self.gp_2b.predict_energy(
            glob_confs, ncores=ncores)
        self.gp_3b.fit_force_and_energy(
            confs, forces - two_body_forces, glob_confs, energies - two_body_energies, ncores=ncores)

    def predict(self, confs, return_std=False, ncores=1):
        """ Predict the forces acting on the central atoms of confs using the
        2- and 3-body GPs. The total force is the sum of the two predictions.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            forces (array): array of force vectors predicted by the GPs
            forces_errors (array): errors associated to the force predictions,
                returned only if return_std is True
        """

        if return_std:
            if self.rep_sig:
                rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                force_2b, std_2b = self.gp_2b.predict(confs, return_std)
                force_2b += rep_forces
            else:
                force_2b, std_2b = self.gp_2b.predict(
                    confs, return_std, ncores=ncores)
            force_3b, std_3b = self.gp_2b.predict(
                confs, return_std, ncores=ncores)
            return force_2b + force_3b, std_2b + std_3b
        else:
            if self.rep_sig:
                rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                return self.gp_2b.predict(confs, return_std, ncores=ncores) + rep_forces + \
                    self.gp_3b.predict(confs, return_std, ncores=ncores)
            else:
                return self.gp_2b.predict(confs, return_std, ncores=ncores) + \
                    self.gp_3b.predict(confs, return_std, ncores=ncores)

    def predict_energy(self, glob_confs, return_std=False, ncores=1):
        """ Predict the local energies of the central atoms of confs using the
        2- and 3-body GPs. The total force is the sum of the two predictions.

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

        if return_std:
            if self.rep_sig:
                rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                force_2b, std_2b = self.gp_2b.predict_energy(
                    glob_confs, return_std, ncores=ncores)
                energy_2b += rep_energies
            else:
                energy_2b, std_2b = self.gp_2b.predict_energy(
                    glob_confs, return_std, ncores=ncoress)
            energy_3b, std_3b = self.gp_2b.predict_energy(
                glob_confs, return_std, ncores=ncores)
            return energy_2b + energy_3b, std_2b + std_3b
        else:
            if self.rep_sig:
                rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                return self.gp_2b.predict_energy(glob_confs, return_std) + rep_energies +\
                    self.gp_3b.predict_energy(
                        glob_confs, return_std, ncores=ncores)
            else:
                return self.gp_2b.predict_energy(glob_confs, return_std, ncores=ncores) + \
                    self.gp_3b.predict_energy(
                        glob_confs, return_std, ncores=ncores)

    def build_grid(self, start, num_2b, num_3b, ncores=1):
        """ Build the mapped 2- and 3-body potentials. 
        Calculates the energy predicted by the GP for two and three atoms at all possible combination 
        of num distances ranging from start to r_cut. The energy for the 3-body mapped grid is 
        calculated only for ``valid`` triplets of atoms, i.e. sets of three distances which 
        form a triangle (this is checked via the triangle inequality). 
        The grid building exploits all the permutation invariances to reduce the number of energy
        calculations needed to fill the grid.
        The computed 2-body energies are stored in an array of values, and a 1D spline interpolation is created.
        The computed 3-body energies are stored in a 3D cube of values, and a 3D spline interpolation is 
        created.
        The total force or local energy can then be calculated for any atom by summing the pairwise and
        triplet contributions of every valid couple and triplet of atoms of which one is always the central one.
        The prediction is done by the ``calculator`` module, which is built to work within 
        the ase python package.

        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 3-body mapped potential
            num_2b (int):number of points to use in the grid of the 2-body mapped potential 
            num_3b (int): number of points to use to generate the list of distances used to
                generate the triplets of atoms for the 2-body mapped potential
            ncores (int): number of CPUs to use to calculate the energy predictions
        """

        dists_2b = np.linspace(start, self.r_cut, num_2b)

        confs = np.zeros((num_2b, 1, 5))
        confs[:, 0, 0] = dists_2b
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element

        grid_data = self.gp_2b.predict_energy(
            confs, ncores=ncores, mapping=True)
        if self.rep_sig:
            grid_data += utility.get_repulsive_energies(
                confs, self.rep_sig, mapping=True)
        grid_2b = interpolation.Spline1D(dists_2b, grid_data)

        # Mapping 3 body part
        dists_3b = np.linspace(start, self.r_cut, num_3b)
        inds, r_ij_x, r_ki_x, r_ki_y = self.generate_triplets(dists_3b)

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

        grid_3b = np.zeros((num_3b, num_3b, num_3b))

        grid_3b[inds] = self.gp_3b.predict_energy(
            confs, ncores=ncores, mapping=True).flatten()

        for ind_i in range(num_3b):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_3b[ind_i, ind_k, ind_j] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_j, ind_i, ind_k] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_j, ind_k, ind_i] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_k, ind_i, ind_j] = grid_3b[ind_i, ind_j, ind_k]
                    grid_3b[ind_k, ind_j, ind_i] = grid_3b[ind_i, ind_j, ind_k]

        grid_3b = interpolation.Spline3D(dists_3b, dists_3b, dists_3b, grid_3b)

        self.grid_2b = grid_2b
        self.grid_3b = grid_3b
        self.grid_num_2b = num_2b
        self.grid_num_3b = num_3b
        self.grid_start = start

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

        ### SAVE THE 2B MODEL ###
        params = {
            'model': self.__class__.__name__,
            'element': self.element,
            'r_cut': self.r_cut,
            'rep_sig': self.rep_sig,
            'fitted': self.gp_2b.fitted,
            'gp_2b': {
                'kernel': self.gp_2b.kernel.kernel_name,
                'n_train': self.gp_2b.n_train,
                'sigma': self.gp_2b.kernel.theta[0],
                'theta': self.gp_2b.kernel.theta[1],
                'noise': self.gp_2b.noise
            },
            'gp_3b': {
                'kernel': self.gp_3b.kernel.kernel_name,
                'n_train': self.gp_3b.n_train,
                'sigma': self.gp_3b.kernel.theta[0],
                'theta': self.gp_3b.kernel.theta[1],
                'noise': self.gp_3b.noise
            },
            'grid_2b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_2b,
                'filename': {}
            } if self.grid_2b else {},
            'grid_3b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_3b,
                'filename': {}
            } if self.grid_3b else {}
        }

        gp_filename_2b = "GP_ker_{p[gp_2b][kernel]}_ntr_{p[gp_2b][n_train]}.npy".format(
            p=params)

        params['gp_2b']['filename'] = gp_filename_2b
        self.gp_2b.save(path / gp_filename_2b)

        if self.grid_2b:
            grid_filename_2b = "GRID_ker_{p[gp_2b][kernel]}_ntr_{p[gp_2b][n_train]}.npz".format(
                p=params)
            print("Saved 2-body grid under name %s" % (grid_filename_2b))
            params['grid_2b']['filename'] = grid_filename_2b
            self.grid_2b.save(path / grid_filename_2b)

        ### SAVE THE 3B MODEL ###
        gp_filename_3b = "GP_ker_{p[gp_3b][kernel]}_ntr_{p[gp_3b][n_train]}.npy".format(
            p=params)

        params['gp_3b']['filename'] = gp_filename_3b
        self.gp_3b.save(path / gp_filename_3b)

        if self.grid_3b:
            grid_filename_3b = "GRID_ker_{p[gp_3b][kernel]}_ntr_{p[gp_3b][n_train]}.npz".format(
                p=params)
            print("Saved 3-body grid under name %s" % (grid_filename_3b))
            params['grid_3b']['filename'] = grid_filename_3b
            self.grid_3b.save(path / grid_filename_3b)

        with open(path / "MODEL_combined_ntr_{p[gp_2b][n_train]}.json".format(p=params), 'w') as fp:
            json.dump(params, fp, indent=4, cls=NpEncoder)

        print("Saved model with name: MODEL_combined_ntr_{p[gp_2b][n_train]}.json".format(p=params))

    @classmethod
    def from_json(cls, path):
        """ Load the model.
        Loads the model, the associated GPs and the mapped potentials, if available.

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
                    params['gp_2b']['sigma'],
                    params['gp_3b']['sigma'],
                    params['gp_2b']['theta'],
                    params['gp_3b']['theta'],
                    params['gp_2b']['noise'],
                    params['rep_sig'])

        gp_filename_2b = params['gp_2b']['filename']
        gp_filename_3b = params['gp_3b']['filename']

        try:
            model.gp_2b.load(directory / gp_filename_2b)
        except:
            warnings.warn("The 2-body GP file is missing")
            pass
        try:
            model.gp_3b.load(directory / gp_filename_3b)
        except:
            warnings.warn("The 3-body GP file is missing")
            pass

        if params['grid_2b']:
            grid_filename_2b = params['grid_2b']['filename']
            model.grid_2b = interpolation.Spline1D.load(
                directory / grid_filename_2b)

            grid_filename_3b = params['grid_3b']['filename']
            model.grid_3b = interpolation.Spline3D.load(
                directory / grid_filename_3b)

            model.grid_start = params['grid_2b']['r_min']
            model.grid_num_2b = params['grid_2b']['r_num']
            model.grid_num_3b = params['grid_3b']['r_num']

        return model

    def save_gp(self, filename_2b, filename_3b):
        """ Saves the GP objects, now obsolete
        """

        warnings.warn('use save and load function', DeprecationWarning)
        self.gp_2b.save(filename_2b)
        self.gp_3b.save(filename_3b)

    def load_gp(self, filename_2b, filename_3b):
        """ Loads the GP objects, now obsolete
        """

        warnings.warn('use save and load function', DeprecationWarning)
        self.gp_2b.load(filename_2b)
        self.gp_3b.load(filename_3b)

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


class CombinedManySpeciesModel(Model):
    """ 2- and 3-body many species model class
    Class managing the Gaussian processes and their mapped counterparts

    Args:
        elements (list): List containing the atomic numbers in increasing order
        r_cut (foat): The cutoff radius used to carve the atomic environments
        sigma_2b (foat): Lengthscale parameter of the 2-body Gaussian process
        sigma_3b (foat): Lengthscale parameter of the 2-body Gaussian process
        theta_2b (float): decay ratio of the cutoff function in the 2-body Gaussian Process
        theta_3b (float): decay ratio of the cutoff function in the 3-body Gaussian Process
        noise (float): noise value associated with the training output data

    Attributes:
        gp_2b (method): The 2-body single species Gaussian Process
        gp_3b (method): The 3-body single species Gaussian Process
        grid_2b (list): Contains the three 2-body two species tabulated potentials, accounting for
            interactions between two atoms of types 0-0, 0-1, and 1-1.     
        grid_2b (list): Contains the three 3-body two species tabulated potentials, accounting for
            interactions between three atoms of types 0-0-0, 0-0-1, 0-1-1, and 1-1-1.  
        grid_start (float): Minimum atomic distance for which the grids are defined (cannot be 0.0)
        grid_num_2b (int):number of points to use in the grid of the 2-body mapped potential 
        grid_num_3b (int): number of points to use to generate the list of distances used to
                generate the triplets of atoms for the 2-body mapped potential  
    """

    def __init__(self, elements, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, rep_sig=1, **kwargs):
        super().__init__()
        self.elements = list(np.sort(elements))
        self.r_cut = r_cut
        self.rep_sig = rep_sig

        kernel_2b = kernels.TwoBodyManySpeciesKernel(
            theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(
            kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodyManySpeciesKernel(
            theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(
            kernel=kernel_3b, noise=noise, **kwargs)

        self.grid_2b, self.grid_3b, self.grid_start, self.grid_num_2b, self.grid_num_3b = {
        }, {}, None, None, None

    def fit(self, confs, forces, ncores=1):
        """ Fit the GP to a set of training forces using a 2- and
        3-body single species force-force kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training forces and the 2-body predictions of force on the 
        training configurations

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """
        hypotetical_model_name = "models/MODEL_ker_TwoBodyManySpecies_ntr_%i.json" %(len(forces))
        try:
            model_2b = models.TwoBodyManySpeciesModel.from_json(hypotetical_model_name)
            self.rep_sig = model_2b.rep_sig
            self.gp_2b = model_2b.gp
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(confs)
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces
            print("Loaded 2-body model to bootstart training")

        except:
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(confs)
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces

            self.gp_2b.fit(confs, forces, ncores=ncores)

        ntr = len(confs)
        two_body_forces = self.gp_2b.predict(confs, ncores=ncores)

        self.gp_3b.fit(confs, forces - two_body_forces, ncores=ncores)

    def fit_energy(self, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body single species energy-energy kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training energies and the 2-body predictions of energies on the 
        training configurations.

        Args:
            glob_confs (list of lists): List of configurations arranged so that
                grouped configurations belong to the same snapshot
            energies (array) : Array containing the total energy of each snapshot
            ncores (int): number of CPUs to use for the gram matrix evaluation
        """
        hypotetical_model_name = "models/MODEL_ker_TwoBodyManySpecies_ntr_%i.json" %(len(energies))
        try:
            model_2b = models.TwoBodyManySpeciesModel.from_json(hypotetical_model_name)
            self.rep_sig = model_2b.rep_sig
            self.gp_2b = model_2b.gp
            if self.rep_sig:
                self.rep_energies = utility.get_repulsive_energies(
                        glob_confs, self.rep_sig)
                energies -= self.rep_energies
            print("Loaded 2-body model to bootstart training")
        
        except:
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(glob_confs)
                self.rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                energies -= self.rep_energies

        self.gp_2b.fit_energy(glob_confs, energies, ncores=1)

        ntr = len(glob_confs)
        two_body_energies = self.gp_2b.predict_energy(
            glob_confs, ncores=ncores)

        self.gp_3b.fit_energy(glob_confs, energies -
                              two_body_energies, ncores=ncores)

    def fit_force_and_energy(self, confs, forces, glob_confs, energies, ncores=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body single species force-force, energy-energy, and energy-forces kernel 
        functions. The 2-body Gaussian process is first fitted, then the 3-body GP 
        is fitted to the difference between the training energies (and forces) and 
        the 2-body predictions of energies (and forces) on the training configurations.

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

        hypotetical_model_name = "models/MODEL_ker_TwoBodyManySpecies_ntr_%i.json" %(len(forces) + len(energies))
        try:
            model_2b = models.TwoBodyManySpeciesModel.from_json(hypotetical_model_name)
            self.rep_sig = model_2b.rep_sig
            self.gp_2b = model_2b.gp
            if self.rep_sig:
                self.rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                energies -= self.rep_energies
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces
            print("Loaded 2-body model to bootstart training")

        except:
            if self.rep_sig:
                self.rep_sig = utility.find_repulstion_sigma(confs)
                self.rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                energies -= self.rep_energies
                self.rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                forces -= self.rep_forces

        self.gp_2b.fit_force_and_energy(
            confs, forces, glob_confs, energies, ncores=ncores)

        two_body_forces = self.gp_2b.predict(confs, ncores=ncores)
        two_body_energies = self.gp_2b.predict_energy(
            glob_confs, ncores=ncores)
        self.gp_3b.fit_force_and_energy(
            confs, forces - two_body_forces, glob_confs, energies - two_body_energies, ncores=ncores)

    def predict(self, confs, return_std=False, ncores=1):
        """ Predict the forces acting on the central atoms of confs using the
        2- and 3-body GPs. The total force is the sum of the two predictions.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework

        Returns:
            forces (array): array of force vectors predicted by the GPs
            forces_errors (array): errors associated to the force predictions,
                returned only if return_std is True
        """

        if return_std:
            if self.rep_sig:
                rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                force_2b, std_2b = self.gp_2b.predict(
                    confs, return_std, ncores=ncores)
                force_2b += rep_forces
            else:
                force_2b, std_2b = self.gp_2b.predict(
                    confs, return_std, ncores=ncores)
            force_3b, std_3b = self.gp_2b.predict(
                confs, return_std, ncores=ncores)
            return force_2b + force_3b, std_2b + std_3b
        else:
            if self.rep_sig:
                rep_forces = utility.get_repulsive_forces(confs, self.rep_sig)
                return self.gp_2b.predict(confs, return_std, ncores=ncores) + rep_forces + \
                    self.gp_3b.predict(confs, return_std, ncores=ncores)
            else:
                return self.gp_2b.predict(confs, return_std, ncores=ncores) + \
                    self.gp_3b.predict(confs, return_std, ncores=ncores)

    def predict_energy(self, glob_confs, return_std=False, ncores=1):
        """ Predict the local energies of the central atoms of confs using the
        2- and 3-body GPs. The total force is the sum of the two predictions.

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

        if return_std:
            if self.rep_sig:
                rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                force_2b, std_2b = self.gp_2b.predict_energy(
                    glob_confs, return_std, ncores=ncores)
                energy_2b += rep_energies
            else:
                energy_2b, std_2b = self.gp_2b.predict_energy(
                    glob_confs, return_std, ncores=ncores)
            energy_3b, std_3b = self.gp_2b.predict_energy(
                glob_confs, return_std, ncores=ncores)
            return energy_2b + energy_3b, std_2b + std_3b
        else:
            if self.rep_sig:
                rep_energies = utility.get_repulsive_energies(
                    glob_confs, self.rep_sig)
                return self.gp_2b.predict_energy(glob_confs, return_std, ncores=ncores) + rep_energies +\
                    self.gp_3b.predict_energy(
                        glob_confs, return_std, ncores=ncores)
            else:
                return self.gp_2b.predict_energy(glob_confs, return_std, ncores=ncores) + \
                    self.gp_3b.predict_energy(
                        glob_confs, return_std, ncores=ncores)

    def build_grid(self, start, num_2b, num_3b, ncores=1):
        """Function used to create the three different 2-body energy grids for 
        atoms of elements 0-0, 0-1, and 1-1, and the four different 3-body energy grids for 
        atoms of elements 0-0-0, 0-0-1, 0-1-1, and 1-1-1. The function calls the
        ``build_grid_3b`` function for each of the 3-body grids to build.

        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 3-body mapped potential
            num (int): number of points to use in the grid of the 2-body mapped potentials
            num_3b (int): number of points to use to generate the list of distances used to
                generate the triplets of atoms for the 3-body mapped potentials
            ncores (int): number of CPUs to use to calculate the energy predictions
        """

        self.grid_start = start
        self.grid_num_2b = num_2b
        self.grid_num_3b = num_2b

        perm_list_2b = list(combinations_with_replacement(self.elements, 2))
        perm_list_3b = list(combinations_with_replacement(self.elements, 3))

        dists_2b = np.linspace(start, self.r_cut, num_2b)
        confs_2b = np.zeros((num_2b, 1, 5))
        confs_2b[:, 0, 0] = dists_2b

        for pair in perm_list_2b:  # in this for loop, predicting then save for each individual one
            confs_2b[:, 0, 3], confs_2b[:, 0,
                                  4] = pair[0], pair[1]

            mapped_energies = self.gp_2b.predict_energy(
                list(confs_2b), ncores=ncores, mapping=True)
            if self.rep_sig:
                mapped_energies += utility.get_repulsive_energies(
                    confs_2b, self.rep_sig, mapping=True)
            self.grid_2b[pair] = interpolation.Spline1D(dists_2b, mapped_energies)


        dists_3b = np.linspace(start, self.r_cut, num_3b)

        for trip in perm_list_3b:

            self.grid_3b[trip] = self.build_grid_3b(
                dists_3b,  trip[0],  trip[1],  trip[2], ncores = ncores)

    def build_grid_3b(self, dists, element_k, element_i, element_j, ncores=1):
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

        # Permutations of elements

        confs[:, :, 3] = element_i  # Central element is always element 1
        confs[:, 0, 4] = element_j  # Element on the x axis is always element 2
        # Element on the xy plane is always element 3
        confs[:, 1, 4] = element_k

        grid_3b = np.zeros((num, num, num))

        grid_3b[inds] = self.gp_3b.predict_energy(
            confs, ncores=ncores, mapping=True).flatten()

        return interpolation.Spline3D(dists, dists, dists, grid_3b)

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

        ### SAVE THE MODEL ###
        params = {
            'model': self.__class__.__name__,
            'elements': self.elements,
            'r_cut': self.r_cut,
            'rep_sig': self.rep_sig,
            'fitted': self.gp_2b.fitted,
            'gp_2b': {
                'kernel': self.gp_2b.kernel.kernel_name,
                'n_train': self.gp_2b.n_train,
                'sigma': self.gp_2b.kernel.theta[0],
                'theta': self.gp_2b.kernel.theta[1],
                'noise': self.gp_2b.noise
            },
            'gp_3b': {
                'kernel': self.gp_3b.kernel.kernel_name,
                'n_train': self.gp_3b.n_train,
                'sigma': self.gp_3b.kernel.theta[0],
                'theta': self.gp_3b.kernel.theta[1],
                'noise': self.gp_3b.noise
            },
            'grid_2b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_2b,
                'filename': {}
            } if self.grid_2b else {},
            'grid_3b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_3b,
                'filename': {}
            } if self.grid_3b else {}
        }

        gp_filename_2b = "GP_ker_{p[gp_2b][kernel]}_ntr_{p[gp_2b][n_train]}.npy".format(
            p=params)

        params['gp_2b']['filename'] = gp_filename_2b
        self.gp_2b.save(path / gp_filename_2b)

        for k, grid in self.grid_2b.items():
            key = '_'.join(str(element) for element in k)
            grid_filename_2b = "GRID_{}_ker_{p[gp_2b][kernel]}_ntr_{p[gp_2b][n_train]}.npz".format(
                key, p=params)
            print("Saved 2-body grid under name %s" % (grid_filename_2b))
            params['grid_2b']['filename'][key] = grid_filename_2b
            grid.save(path / grid_filename_2b)

        ### SAVE THE 3B MODEL ###
        gp_filename_3b = "GP_ker_{p[gp_3b][kernel]}_ntr_{p[gp_3b][n_train]}.npy".format(
            p=params)

        params['gp_3b']['filename'] = gp_filename_3b
        self.gp_3b.save(path / gp_filename_3b)

        for k, grid in self.grid_3b.items():
            key = '_'.join(str(element) for element in k)
            grid_filename_3b = "GRID_{}_ker_{p[gp_3b][kernel]}_ntr_{p[gp_3b][n_train]}.npz".format(
                key, p=params)
            print("Saved 3-body grid under name %s" % (grid_filename_3b))
            params['grid_3b']['filename'][key] = grid_filename_3b
            grid.save(path / grid_filename_3b)

        with open(path / "MODEL_combined_ntr_{p[gp_2b][n_train]}.json".format(p=params), 'w') as fp:
            json.dump(params, fp, indent=4, cls=NpEncoder)

        print("Saved model with name: MODEL_combined_ntr_{p[gp_2b][n_train]}.json".format(p=params))

    @classmethod
    def from_json(cls, path):
        """ Load the model.
        Loads the model, the associated GPs and the mapped potentials, if available.

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
                    params['gp_2b']['sigma'],
                    params['gp_3b']['sigma'],
                    params['gp_2b']['theta'],
                    params['gp_3b']['theta'],
                    params['gp_2b']['noise'],
                    params['rep_sig'])

        gp_filename_2b = params['gp_2b']['filename']
        gp_filename_3b = params['gp_3b']['filename']

        try:
            model.gp_2b.load(directory / gp_filename_2b)
        except:
            warnings.warn("The 2-body GP file is missing")
            pass
        try:
            model.gp_3b.load(directory / gp_filename_3b)
        except:
            warnings.warn("The 3-body GP file is missing")
            pass

        if params['grid_2b']:
            for key, grid_filename_2b in params['grid_2b']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid_2b[k] = interpolation.Spline1D.load(
                    directory / grid_filename_2b)

            for key, grid_filename_3b in params['grid_3b']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid_3b[k] = interpolation.Spline3D.load(
                    directory / grid_filename_3b)

            model.grid_start = params['grid_2b']['r_min']
            model.grid_num_2b = params['grid_2b']['r_num']
            model.grid_num_3b = params['grid_3b']['r_num']

        return model

    def save_gp(self, filename_2b, filename_3b):
        """ Saves the GP objects, now obsolete
        """

        self.gp_2b.save(filename_2b)
        self.gp_3b.save(filename_3b)

    def load_gp(self, filename_2b, filename_3b):
        """ Loads the GP objects, now obsolete
        """

        self.gp_2b.load(filename_2b)
        self.gp_3b.load(filename_3b)

    @staticmethod
    def generate_triplets_all(dists):
        """ Generate a list of all valid triplets.
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
