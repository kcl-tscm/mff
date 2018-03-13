# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 14:55:57 2018

@author: k1506329
"""

from ase.io import read
import numpy as np
from ase.calculators.neighborlist import NeighborList


def carve_from_snapshot(atoms, atoms_ind, cutoffs):
  # See if there are forces and energies, get them for the chosen atoms

  cell= atoms.get_cell()
  confs = []
  errvalue = 0
  try:
    forces = atoms.get_array('force')[atoms_ind]
  except KeyError:
    print('Forces in the xyz file are not present, or are not called force')
    forces = None
    errvalue += 1
  try:
    energy = atoms.get_array('energy')
  except KeyError:
    print('Energy in the xyz file is not present, or is not called energy')
    energy = None
    errvalue += 1
  if errvalue == 2 :
    print('Cannot find energy or force values in the xyz file, shitting down')
    quit()

  nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False, bothways=True)
  nl.build(atoms)

  # Build local configurations for every indexed atom
  for i in atoms_ind:
    indices, offsets = nl.get_neighbors(i)
    offsets = np.dot(offsets, cell)
    conf = np.zeros((len(indices), 5))

    for k, (a2, offset) in enumerate(zip(indices, offsets)):
      d = atoms.positions[a2] + offset - atoms.positions[i]
      conf[k, :3] = d
      conf[k, 4] = atoms.get_atomic_numbers()[a2]

    conf[:, 3] = atoms.get_atomic_numbers()[i]
    confs.append(conf)
    
  return(confs, forces, energy)
  
  
def carve_confs(filename, r_cut, n_data = 3000):

  confs = []
  forces  = []
  energies = []

  # Open file and get number of atoms and steps
  f = open(filename, 'r')
  num_lines = 1 + sum(1 for line in f)
  atoms = read(filename, index=':', format = 'extxyz')

  # Get the atomic number of each atom in the trajectory file
  atom_number_list = []
  for i in np.arange(len(atoms)):
      atom_number_list.append(atoms[i].get_atomic_numbers())
  flat_atom_number = np.concatenate(atom_number_list).ravel()
  elements, elements_count = np.unique(flat_atom_number, return_counts=True)

  # Calculate the ratios of occurrence of central atoms based on their atomic number
  ratios = np.zeros(len(elements))
  for i in np.arange(len(elements)):
    ratios[i] = np.sqrt(float(elements_count[i]))
  ratios = ratios/(np.sum(ratios))

  # Obtain the indices of the atoms we want in the final database from a linspace on the the flattened array
  indices = []
  for i in np.arange(len(elements)):
    indices.append(np.linspace(0, elements_count[i], int(ratios[i]*n_data) - 1))
    indices[i] = map(int, indices[i])  # This is the n-th atom of that type you have to pick

  # Go through each trajectory step and find where the chosen indexes for all different elements are
  element_ind_count = np.zeros(len(elements))
  element_ind_count_prev = np.zeros(len(elements))
  for j in np.arange(len(atoms)):
    print("Reading traj step %i" %(j))
    this_ind = []
    for k in np.arange(len(elements)):
      count_el_atoms = sum(atom_number_list[j] ==  elements[k])
      element_ind_count[k] += count_el_atoms
      temp_ind = np.asarray([x for i,x in enumerate(indices[k] - element_ind_count_prev[k])
                       if (0 <= x < count_el_atoms)])
      temp_ind = temp_ind.astype(int)
      this_ind.append((np.where(atom_number_list[j] == elements[k]))[0][temp_ind])
      element_ind_count_prev[k] += count_el_atoms
    this_ind = np.concatenate(this_ind).ravel()

    # Call the carve_from_snapshot function on the chosen atoms
    if len(this_ind) > 0:
      cutoffs = np.ones(len(atom_number_list[j])) * r_cut / 2.
      this_conf, this_force, this_energy = carve_from_snapshot(atoms[j], this_ind, cutoffs)
      confs.append(this_conf)
      forces.append(this_force)
      energies.append(this_energy)

  # Reshape everything so that confs is a list of numpy arrays, forces is a numpy array and energies is a numpy array
  confs = [item for sublist in confs for item in sublist]
  forces = [item for sublist in forces for item in sublist]

  forces = np.asarray(forces)
  energies = np.asarray(energies)

  np.save("confs_cut=%.2f.npy" % (r_cut), confs)
  np.save("forces_cut=%.2f.npy" % (r_cut), forces)
  np.save("energies_cut=%.2f.npy" % (r_cut), energies)
  lens = []
  for i in np.arange(len(confs)):
    lens.append(len(confs[i]))

  print("Maximum number of atoms in a configuration", max(lens))
  print("Minimum number of atoms in a configuration", min(lens))
  print("Average number of atoms in a configuration", np.mean(lens))

  return elements

a = carve_confs("C_a/movie.xyz", 3.7)
