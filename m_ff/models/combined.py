# -*- coding: utf-8 -*-
"""
Combined Model
==============

Module that uses 2- and 3-body kernels to do Guassian process regression, 
and to build 2- and 3-body mapped potentials.
The model has to be first defined, then the Gaussian processes must be
trained using training configurations and forces (and/or energies).
Once a model has been trained, it can be used to predict forces 
(and/or energies) on unknonwn atomic configurations.
A trained Gaussian process can then be mapped onto a tabulated 2-body
potential  and a tabultaed 3-body potential via the ``build grid`` function call.
A mapped model can be thensaved, loaded and used to run molecular 
dynamics simulations via the calculator module.
These mapped potentials retain the accuracy of the GP used to build them,
while speeding up the calculations by a factor of 10^4 in typical scenarios.

Example:

    Basic usage::

        from m_ff import models
        mymodel = models.CombinedSingleSpecies(atomic_number, cutoff_radius,
                        sigma_2b, sigma_3b, sigma_2b, theta_3b, noise)
        mymodel.fit(training_confs, training_forces)
        forces = mymodel.predict(test_configurations)
        mymodel.build_grid(grid_start, num_2b)
        mymodel.save("thismodel.json")
        mymodel = models.CombinedSingleSpecies.from_json("thismodel.json")

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html
"""

import json
import numpy as np

from pathlib import Path

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation

from .base import Model
import logging
import warnings

logger = logging.getLogger(__name__)


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

    def __init__(self, element, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, **kwargs):
        super().__init__()
        self.element = element
        self.r_cut = r_cut

        kernel_2b = kernels.TwoBodySingleSpeciesKernel(theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(kernel=kernel_3b, noise=noise, **kwargs)

        self.grid_2b, self.grid_3b, self.grid_start, self.grid_num = None, None, None, None

    def fit(self, confs, forces, nnodes=1):
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
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp_2b.fit(confs, forces, nnodes)

        ntr = len(confs)
        two_body_forces = np.zeros((ntr, 3))
        for i in np.arange(ntr):
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit(confs, forces - two_body_forces, nnodes)

    def fit_energy(self, confs, energies, nnodes=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body single species energy-energy kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training energies and the 2-body predictions of energies on the 
        training configurations.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp_2b.fit_energy(confs, energies, nnodes)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit_energy(confs, energies - two_body_energies, nnodes)

    def fit_force_and_energy(self, confs, forces, energies, nnodes=1):
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
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp_2b.fit_force_and_energy(confs, forces, energies, nnodes)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        two_body_forces = np.zeros((ntr, 3))

        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))
        self.gp_3b.fit_force_and_energy(confs, forces - two_body_forces, energies - two_body_energies, nnodes)

    def predict(self, confs, return_std=False):
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
            force_2b, std_2b = self.gp_2b.predict(confs, return_std)
            force_3b, std_3b = self.gp_2b.predict(confs, return_std)
            return force_2b + force_3b, std_2b + std_3b
        else:
            return self.gp_2b.predict(confs, return_std) + \
                   self.gp_3b.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        """ Predict the local energies of the central atoms of confs using the
        2- and 3-body GPs. The total force is the sum of the two predictions.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework
            
        Returns:
            energies (array): array of force vectors predicted by the GPs
            energies_errors (array): errors associated to the energies predictions,
                returned only if return_std is True
        """

        if return_std:
            energy_2b, std_2b = self.gp_2b.predict_energy(confs, return_std)
            energy_3b, std_3b = self.gp_2b.predict_energy(confs, return_std)
            return energy_2b + energy_3b, std_2b + std_3b
        else:
            return self.gp_2b.predict_energy(confs, return_std) + \
                   self.gp_3b.predict_energy(confs, return_std)

    def build_grid(self, start, num_2b, num_3b, nnodes=1):
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
            nnodes (int): number of CPUs to use to calculate the energy predictions
        """

        dists_2b = np.linspace(start, self.r_cut, num_2b)

        confs = np.zeros((num_2b, 1, 5))
        confs[:, 0, 0] = dists_2b
        confs[:, 0, 3], confs[:, 0, 4] = self.element, self.element

        grid_data = self.gp_2b.predict_energy(confs)
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
        confs[:, 0, 4] = self.element  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element  # Element on the xy plane is always element 3

        grid_3b = np.zeros((num_3b, num_3b, num_3b))

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)
            import sys
            sys.setrecursionlimit(1000000)
            logger.info('Using %i cores for the mapping' % (nnodes))
            pool = ProcessingPool(nodes=nnodes)
            splitind = np.zeros(nnodes + 1)
            factor = (n + (nnodes - 1)) / nnodes
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
            result = np.array(pool.map(self.gp_3b.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_3b[inds] = result

        else:
            grid_3b[inds] = self.gp_3b.predict_energy(confs).flatten()

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

    def save_combined(self, path):
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

        ### SAVE THE 2B MODEL ###
        params = {
            'model': self.__class__.__name__,
            'element': self.element,
            'r_cut': self.r_cut,
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
                'r_num': self.grid_num_2b
            } if self.grid_2b else {}
            ,
            'grid_3b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_3b
            } if self.grid_3b else {}
        }

        gp_filename_2b = "{}_gp_ker_2_ntr_{p[gp_2b][n_train]}.npy".format(prefix, p=params)

        params['gp_2b']['filename'] = gp_filename_2b
        self.gp_2b.save(directory / gp_filename_2b)

        if self.grid_2b:
            grid_filename_2b = '{}_grid_2b_num_{p[grid_2b][r_num]}.npz'.format(prefix, p=params)
            print("Saved 2-body grid under name %s" % (grid_filename_2b))
            params['grid_2b']['filename'] = grid_filename_2b
            self.grid_2b.save(directory / grid_filename_2b)

        ### SAVE THE 3B MODEL ###
        gp_filename_3b = "{}_gp_ker_3_ntr_{p[gp_3b][n_train]}.npy".format(prefix, p=params)

        params['gp_3b']['filename'] = gp_filename_3b
        self.gp_3b.save(directory / gp_filename_3b)

        if self.grid_3b:
            grid_filename_3b = '{}_grid_3b_num_{p[grid_3b][r_num]}.npz'.format(prefix, p=params)
            print("Saved 3-body grid under name %s" % (grid_filename_3b))
            params['grid_3b']['filename'] = grid_filename_3b
            self.grid_3b.save(directory / grid_filename_3b)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)

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
                    params['gp_2b']['noise'])

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
            model.grid_2b = interpolation.Spline1D.load(directory / grid_filename_2b)

            grid_filename_3b = params['grid_3b']['filename']
            model.grid_3b = interpolation.Spline3D.load(directory / grid_filename_3b)

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

        d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Utilizing permutation invariance
        inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y


class CombinedTwoSpeciesModel(Model):
    """ 2- and 3-body two species model class
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

    def __init__(self, elements, r_cut, sigma_2b, sigma_3b, theta_2b, theta_3b, noise, **kwargs):
        super().__init__()
        self.elements = elements
        self.r_cut = r_cut

        kernel_2b = kernels.TwoBodyTwoSpeciesKernel(theta=[sigma_2b, theta_2b, r_cut])
        self.gp_2b = gp.GaussianProcess(kernel=kernel_2b, noise=noise, **kwargs)

        kernel_3b = kernels.ThreeBodyTwoSpeciesKernel(theta=[sigma_3b, theta_3b, r_cut])
        self.gp_3b = gp.GaussianProcess(kernel=kernel_3b, noise=noise, **kwargs)

        self.grid_2b, self.grid_3b, self.grid_start, self.grid_num_2b, self.grid_num_3b = {}, {}, None, None, None

    def fit(self, confs, forces, nnodes=1):
        """ Fit the GP to a set of training forces using a 2- and
        3-body two species force-force kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training forces and the 2-body predictions of force on the 
        training configurations

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp_2b.fit(confs, forces, nnodes)

        ntr = len(confs)
        two_body_forces = np.zeros((ntr, 3))
        for i in np.arange(ntr):
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit(confs, forces - two_body_forces, nnodes)

    def fit_energy(self, confs, energies, nnodes=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body two species energy-energy kernel functions. The 2-body Gaussian
        process is first fitted, then the 3-body GP is fitted to the difference
        between the training energies and the 2-body predictions of energies on the 
        training configurations.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp_2b.fit_energy(confs, energies, nnodes)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))

        self.gp_3b.fit_energy(confs, energies - two_body_energies, nnodes)

    def fit_force_and_energy(self, confs, forces, energies, nnodes=1):
        """ Fit the GP to a set of training energies using a 2- and
        3-body two species force-force, energy-energy, and energy-forces kernel 
        functions. The 2-body Gaussian process is first fitted, then the 3-body GP 
        is fitted to the difference between the training energies (and forces) and 
        the 2-body predictions of energies (and forces) on the training configurations.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp_2b.fit_force_and_energy(confs, forces, energies, nnodes)

        ntr = len(confs)
        two_body_energies = np.zeros(ntr)
        two_body_forces = np.zeros((ntr, 3))

        for i in np.arange(ntr):
            two_body_energies[i] = self.gp_2b.predict_energy(np.reshape(confs[i], (1, len(confs[i]), 5)))
            two_body_forces[i] = self.gp_2b.predict(np.reshape(confs[i], (1, len(confs[i]), 5)))
        self.gp_3b.fit_force_and_energy(confs, forces - two_body_forces, energies - two_body_energies, nnodes)

    def predict(self, confs, return_std=False):
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
            force_2b, std_2b = self.gp_2b.predict(confs, return_std)
            force_3b, std_3b = self.gp_2b.predict(confs, return_std)
            return force_2b + force_3b, std_2b + std_3b
        else:
            return self.gp_2b.predict(confs, return_std) + \
                   self.gp_3b.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
        """ Predict the local energies of the central atoms of confs using the
        2- and 3-body GPs. The total force is the sum of the two predictions.

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            return_std (bool): if True, returns the standard deviation 
                associated to predictions according to the GP framework
            
        Returns:
            energies (array): array of force vectors predicted by the GPs
            energies_errors (array): errors associated to the energies predictions,
                returned only if return_std is True
        """

        if return_std:
            energy_2b, std_2b = self.gp_2b.predict_energy(confs, return_std)
            energy_3b, std_3b = self.gp_2b.predict_energy(confs, return_std)
            return energy_2b + energy_3b, std_2b + std_3b
        else:
            return self.gp_2b.predict_energy(confs, return_std) + \
                   self.gp_3b.predict_energy(confs, return_std)

    def build_grid(self, start, num_2b, num_3b, nnodes=1):
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
            nnodes (int): number of CPUs to use to calculate the energy predictions
        """

        self.grid_start = start
        self.grid_num_2b = num_2b
        self.grid_num_3b = num_2b

        dists = np.linspace(start, self.r_cut, num_2b)

        confs = np.zeros((num_2b, 1, 5))
        confs[:, 0, 0] = dists

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[0], self.elements[0]
        grid_0_0_data = self.gp_2b.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[0], self.elements[1]
        grid_0_1_data = self.gp_2b.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.elements[1], self.elements[1]
        grid_1_1_data = self.gp_2b.predict_energy(confs)

        self.grid_2b[(0, 0)] = interpolation.Spline1D(dists, grid_0_0_data)
        self.grid_2b[(0, 1)] = interpolation.Spline1D(dists, grid_0_1_data)
        self.grid_2b[(1, 1)] = interpolation.Spline1D(dists, grid_1_1_data)

        dists = np.linspace(start, self.r_cut, num_3b)

        grid_0_0_0 = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[0], nnodes)
        grid_0_0_1 = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[1], nnodes)
        grid_0_1_1 = self.build_grid_3b(dists, self.elements[0], self.elements[1], self.elements[1], nnodes)
        grid_1_1_1 = self.build_grid_3b(dists, self.elements[1], self.elements[1], self.elements[1], nnodes)

        self.grid_3b[(0, 0, 0)] = grid_0_0_0
        self.grid_3b[(0, 0, 1)] = grid_0_0_1
        self.grid_3b[(0, 1, 1)] = grid_0_1_1
        self.grid_3b[(1, 1, 1)] = grid_1_1_1

    def build_grid_3b(self, dists, element_k, element_i, element_j, nnodes):
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
            nnodes (int): number of CPUs to use when computing the triplet local energies
            
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
        confs[:, 1, 4] = element_k  # Element on the xy plane is always element 3

        grid_3b = np.zeros((num, num, num))

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)
            import sys
            sys.setrecursionlimit(1000000)
            logger.info('Using %i cores for the mapping' % (nnodes))
            pool = ProcessingPool(nodes=nnodes)
            splitind = np.zeros(nnodes + 1)
            factor = (n + (nnodes - 1)) / nnodes
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
            result = np.array(pool.map(self.gp_3b.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_3b[inds] = result

        else:
            grid_3b[inds] = self.gp_3b.predict_energy(confs).flatten()

        return interpolation.Spline3D(dists, dists, dists, grid_3b)

    def save_combined(self, path):
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

        ### SAVE THE 2B MODEL ###
        params = {
            'model': self.__class__.__name__,
            'elements': self.elements,
            'r_cut': self.r_cut,
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
                'r_num': self.grid_num_2b
            } if self.grid_2b else {}
            ,
            'grid_3b': {
                'r_min': self.grid_start,
                'r_num': self.grid_num_3b
            } if self.grid_3b else {}
        }

        gp_filename_2b = "{}_gp_ker_2_ntr_{p[gp_2b][n_train]}.npy".format(prefix, p=params)

        params['gp_2b']['filename'] = gp_filename_2b
        self.gp_2b.save(directory / gp_filename_2b)

        params['grid_2b']['filename'] = {}
        for k, grid in self.grid_2b.items():
            key = '_'.join(str(element) for element in k)
            grid_filename_2b = '{}_grid_{}_num_{p[grid_2b][r_num]}.npz'.format(prefix, key, p=params)
            print("Saved 2-body grid under name %s" % (grid_filename_2b))
            params['grid_2b']['filename'][key] = grid_filename_2b
            self.grid_2b[k].save(directory / grid_filename_2b)

        ### SAVE THE 3B MODEL ###
        gp_filename_3b = "{}_gp_ker_3_ntr_{p[gp_3b][n_train]}.npy".format(prefix, p=params)

        params['gp_3b']['filename'] = gp_filename_3b
        self.gp_3b.save(directory / gp_filename_3b)

        params['grid_3b']['filename'] = {}
        for k, grid in self.grid_3b.items():
            key = '_'.join(str(element) for element in k)
            grid_filename_3b = '{}_grid_{}_num_{p[grid_3b][r_num]}.npz'.format(prefix, key, p=params)
            print("Saved 3-body grid under name %s" % (grid_filename_3b))
            params['grid_3b']['filename'][key] = grid_filename_3b
            self.grid_3b[k].save(directory / grid_filename_3b)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)

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
                    params['gp_2b']['noise'])

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
                model.grid_2b[k] = interpolation.Spline1D.load(directory / grid_filename_2b)

            for key, grid_filename_3b in params['grid_3b']['filename'].items():
                k = tuple(int(ind) for ind in key.split('_'))
                model.grid_3b[k] = interpolation.Spline3D.load(directory / grid_filename_3b)

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

        d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Element on the x axis
        r_ij_x = d_ij[inds]

        # Element on the xy plane
        r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])

        # using abs to avoid numerical error near to 0
        r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

        return inds, r_ij_x, r_ki_x, r_ki_y
