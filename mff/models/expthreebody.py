# -*- coding: utf-8 -*-


import json
import numpy as np
import warnings

from pathlib import Path
from mff import gp
from mff import kernels
from mff import interpolation
from mff.models.base import Model


class ExpThreeBodySingleSpeciesModel(Model):
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

    def __init__(self, element, r_cut, sigma, theta, noise, chi, **kwargs):
        super().__init__()

        self.element = element
        self.r_cut = r_cut

        kernel = kernels.ExpThreeBodySingleSpeciesKernel(theta=[sigma, theta, r_cut, chi])
        self.gp = gp.GaussianProcess(kernel=kernel, noise=noise, **kwargs)

        self.grid, self.grid_start, self.grid_num = None, None, None

    def fit(self, confs, forces, nnodes=1):
        """ Fit the GP to a set of training forces using a 
        2-body single species force-force kernel

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit(confs, forces, nnodes)

    def fit_energy(self, confs, energies, nnodes=1):
        """ Fit the GP to a set of training energies using a 
        2-body single species energy-energy kernel

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation
        """

        self.gp.fit_energy(confs, energies, nnodes)

    def fit_force_and_energy(self, confs, forces, energies, nnodes=1):
        """ Fit the GP to a set of training forces and energies using 
        2-body single species force-force, energy-force and energy-energy kernels

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
        
    def update_force(self, confs, forces, nnodes=1):
        """ Update a fitted GP with a set of forces and using 
        2-body single species force-force kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            forces (array) : Array containing the vector forces on 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation

        """

        self.gp.fit_update(confs, forces, nnodes)
        
    def update_energy(self, confs, energies, nnodes=1):
        """ Update a fitted GP with a set of energies and using 
        2-body single species energy-energy kernels

        Args:
            confs (list): List of M x 5 arrays containing coordinates and
                atomic numbers of atoms within a cutoff from the central one
            energies (array) : Array containing the scalar local energies of 
                the central atoms of the training configurations
            nnodes (int): number of CPUs to use for the gram matrix evaluation

        """

        self.gp.fit_update_energy(confs, energies, nnodes)
        
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
