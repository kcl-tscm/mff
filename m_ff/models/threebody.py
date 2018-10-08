# -*- coding: utf-8 -*-
"""
Three Body Model
================

Module containing the ThreeBodySingleSpecies and 
ThreeBodyTwoSpecies classes, which are used to handle the Gaussian 
process regression and the mapping algorithm used to build M-FFs.
The model has to be first defined, then the Gaussian process must be
trained using training configurations and forces (and/or local energies).
Once a model has been trained, it can be used to predict forces 
(and/or energies) on unknonwn atomic configurations.
A trained Gaussian process can then be mapped onto a tabulated 3-body
potential via the ``build grid`` function call. A mapped model can be then
saved, loaded and used to run molecular dynamics simulations via the
calculator module.
These mapped potentials retain the accuracy of the GP used to build them,
while speeding up the calculations by a factor of 10^4 in typical scenarios.

Example:

    >>> from m_ff import models
    >>> mymodel = models.ThreeBodySingleSpecies(atomic_number, cutoff_radius, sigma, theta, noise)
    >>> mymodel.fit(training_confs, training_forces)
    >>> forces = mymodel.predict(test_configurations)
    >>> mymodel.build_grid(grid_start, num_3b)
    >>> mymodel.save("thismodel.json")
    >>> mymodel = models.CombinedSingleSpecies.from_json("thismodel.json")

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import json
import numpy as np
import warnings

from m_ff import gp
from m_ff import kernels
from m_ff import interpolation
from pathlib import Path
from m_ff.models.base import Model


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

        kernel = kernels.ThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

    def fit(self, confs, forces, nnodes = 1):
        """ Fit the GP to a set of training forces using a 
        3-body single species force-force kernel function

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """
        
        self.gp.fit(confs, forces, nnodes)

    def fit_energy(self, confs, energies, nnodes = 1):
        """ Fit the GP to a set of training energies using a 
        3-body single species energy-energy kernel function

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """
        
        self.gp.fit_energy(confs, energies, nnodes)

    def fit_force_and_energy(self, confs, forces, energies, nnodes = 1):
        """ Fit the GP to a set of training forces and energies using 
        3-body single species force-force, energy-force and energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """
        
        self.gp.fit_force_and_energy(confs, forces, energies, nnodes)

    def predict(self, confs, return_std=False):
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
        
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
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
        
        return self.gp.predict_energy(confs, return_std)

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

    def build_grid(self, start, num, nnodes = 1):
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
            nnodes (int): number of CPUs to use to calculate the energy predictions
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
        confs[:, 0, 4] = self.element  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element  # Element on the xy plane is always element 3

        grid_data = np.zeros((num, num, num))

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)
            import sys
            sys.setrecursionlimit(1000000)
            print('Using %i cores for the mapping' % (nnodes))
            pool = ProcessingPool(nodes=nnodes)
            splitind = np.zeros(nnodes + 1)
            factor = (n + (nnodes - 1)) / nnodes
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
            result = np.array(pool.map(self.gp.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_data[inds] = result
            
        else:
            grid_data[inds] = self.predict_energy(confs).flatten()
            
        for ind_i in range(num):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    grid_data[ind_i, ind_k, ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_i, ind_k] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_j, ind_k, ind_i] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_i, ind_j] = grid_data[ind_i, ind_j, ind_k]
                    grid_data[ind_k, ind_j, ind_i] = grid_data[ind_i, ind_j, ind_k]

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
            'elements': self.element,
            'r_cut': self.r_cut,
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {
                'r_min': self.grid_start,
                'r_num': self.grid_num
            } if self.grid else {}
        }

        gp_filename = "{}_gp_ker_3_ntr_{p[gp][n_train]}".format(prefix, p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(directory / gp_filename)

        if self.grid:
            grid_filename = '{}_grid_num_{p[grid][r_num]}'.format(prefix, p=params)

            params['grid']['filename'] = grid_filename
            self.grid.save(directory / grid_filename)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)


class ThreeBodyTwoSpeciesModel(Model):
    """ 3-body two species model class
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
        self.elements = elements
        self.r_cut = r_cut

        kernel = kernels.ThreeBodyTwoSpeciesKernel(theta=[sigma, theta, r_cut])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = {}, None, None

    def fit(self, confs, forces, nnodes = 1):
        """ Fit the GP to a set of training forces using a 
        3-body two species force-force kernel function

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """
        
        self.gp.fit(confs, forces, nnodes)

    def fit_energy(self, confs, forces, nnodes = 1):
        """ Fit the GP to a set of training energies using a 
        3-body two species energy-energy kernel function

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """
        
        self.gp.fit_energy(confs, forces, nnodes)

    def fit_force_and_energy(self, confs, forces, energies, nnodes = 1):
        """ Fit the GP to a set of training forces and energies using 
        3-body two species force-force, energy-force and energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """
        
        self.gp.fit_force_and_energy(confs, forces, energies, nnodes)

    def predict(self, confs, return_std=False):
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
        
        return self.gp.predict(confs, return_std)

    def predict_energy(self, confs, return_std=False):
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
        
        return self.gp.predict_energy(confs, return_std)

    def save_gp(self, filename):
        """ Saves the GP object, now obsolete
        """
    
        self.gp.save(filename)

    def load_gp(self, filename):
        """ Loads the GP object, now obsolete
        """
        
        self.gp.load(filename)

    def build_grid(self, start, num, nnodes = 1):
        """Function used to create the four different 3-body energy grids for 
        atoms of elements 0-0-0, 0-0-1, 0-1-1, and 1-1-1. The function calls the
        ``build_grid_3b`` function for each of those combinations of elements.
        
        Args:
            start (float): smallest interatomic distance for which the energy is predicted
                by the GP and stored inn the 3-body mapped potential
            num (int): number of points to use to generate the list of distances used to
                generate the triplets of atoms for the mapped potential
            nnodes (int): number of CPUs to use to calculate the energy predictions
        """
        
        self.grid_start = start
        self.grid_num = num

        dists = np.linspace(start, self.r_cut, num)
        self.grid[(0, 0, 0)] = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[0], nnodes)
        self.grid[(0, 0, 1)] = self.build_grid_3b(dists, self.elements[0], self.elements[0], self.elements[1], nnodes)
        self.grid[(0, 1, 1)] = self.build_grid_3b(dists, self.elements[0], self.elements[1], self.elements[1], nnodes)
        self.grid[(1, 1, 1)] = self.build_grid_3b(dists, self.elements[1], self.elements[1], self.elements[1], nnodes)

    def build_grid_3b(self, dists, element_i, element_j, element_k, nnodes):
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

        confs[:, :, 3] = element_i  # Central element is always element 1
        confs[:, 0, 4] = element_j  # Element on the x axis is always element 2
        confs[:, 1, 4] = element_k  # Element on the xy plane is always element 3
        
        grid_3b = np.zeros((num, num, num))

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)
            import sys
            sys.setrecursionlimit(1000000)
            print('Using %i cores for the mapping' % (nnodes))
            pool = ProcessingPool(nodes=nnodes)
            splitind = np.zeros(nnodes + 1)
            factor = (n + (nnodes - 1)) / nnodes
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
            result = np.array(pool.map(self.gp.predict_energy, clist))
            result = np.concatenate(result).flatten()
            grid_3b[inds] = result
            
        else:
            grid_3b[inds] = self.gp.predict_energy(confs).flatten()

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
            'gp': {
                'kernel': self.gp.kernel.kernel_name,
                'n_train': self.gp.n_train,
                'sigma': self.gp.kernel.theta[0],
                'theta': self.gp.kernel.theta[1],
                'noise': self.gp.noise
            },
            'grid': {
                'r_min': self.grid_start,
                'r_num': self.grid_num
            } if self.grid else {}
        }

        gp_filename = "{}_gp_ker_3_ntr_{p[gp][n_train]}".format(prefix, p=params)

        params['gp']['filename'] = gp_filename
        self.gp.save(directory / gp_filename)

        params['grid']['filename'] = {}
        for k, grid in self.grid.items():
            key = '_'.join(str(element) for element in k)
            grid_filename = '{}_grid_{}_num_{p[grid][r_num]}'.format(prefix, key, p=params)

            params['grid']['filename'][key] = grid_filename
            self.grid[k].save(directory / grid_filename)

        with open(directory / '{}.json'.format(prefix), 'w') as fp:
            json.dump(params, fp, indent=4)

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
                model.grid[k] = interpolation.Spline3D.load(directory / grid_filename)
            
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
            np.hstack([np.random.randn(4, 3), np.random.choice(elements, size=(4, 2))]),
            np.hstack([np.random.randn(5, 3), np.random.choice(elements, size=(5, 2))])
        ])

        forces = np.random.randn(2, 3)
        r_cut, sigma, theta, noise = 2., 3., 4., 5.

        filename = Path() / 'test_model'

        m = ThreeBodyTwoSpeciesModel(elements, r_cut, sigma, theta, noise)
        print(m)

        m.fit(confs, forces)
        print(m)

        m.build_grid(1., 10)
        print(m)

        m.save(filename)


    test_three_body_single_species_model()
    # test_three_body_two_species_model()
