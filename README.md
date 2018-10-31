# Machine learning nonparametric force fields (MFFs)

To read the full documentation check https://mff.readthedocs.io/en/latest/

## Table of Contents

- [Background on MFFs](#background)
- [Install](#install)
- [Usage](#usage)
- [Examples](#examples)
- [Maintainers](#maintainers)
- [References](#references)

## Background on MFF

The MFF package uses Gaussian process regression to extract non-parametric 2- and 3- body force fields from ab-initio calculations.
For a detailed description of the theory behind Gaussian process regression to predict forces and/or energies, and an explanation of the mapping technique used, please refer to [1].

For an example use of the M-FF package to build 3-body force fields for Ni nanoclusters, please see [2].

## Install

To install from source, uncompress the source files and, from the directory containing `setup.py`, run the following command:
    
    python setup.py install

Or, to build in place, run:
    
    python setup.py build_ext --inplace

If you build in place, you will also need to add your eqtools folder to your PYTHONPATH shell variable:
    
    export PYTHONPATH=$PYTHONPATH:/path/to/where/you/put/

## Usage

Description on how to use the package.

```py
import mff

```

## Examples
This shows a simple example use of the package, starting from an .xzy trajectory obtained through ab-initio methods and
finishing with a 3-body MFF which can be used to run fast and accurate simulations within the python ASE framework.

First, import the ab-initio trajectory file and create local atomic environments, force arrays and energy arrays:

```py
import numpy as np
from ase.io import read
from mff.configurations import carve_confs
traj = read(filename, format='extxyz')
r_cut = 5.0  # Angstrom
n_data = 5000
elements, confs, forces, energies = carve_confs(traj, r_cut, n_data)

```

Then, separate into training and test set:

```py
ntr, ntest = 200, 100
ind = np.arange(numconfs)
ind_tot = np.random.choice(ind, size=ntr + ntest, replace=False)
tr_confs, tr_forces = confs[ind_tot[:ntr]], forces[ind_tot[:ntr]]
tst_confs, tst_forces = confs[ind_tot[ntr:]], forces[ind_tot[ntr:]]

```

Create the model and fit it to the forces:

```py
from mff import models
sigma_2b, theta_2b, sigma_3b, theta_3b, noise = 0.2, 0.05, 1.0, 0.05, 0.001
mymodel = models.CombinedSingleSpecies(elements, r_cut, sigma_2b, theta_2b, sigma_3b, theta_3b, noise)
mymodel.fit(tr_confs, tr_forces)

```

Test the model on a separate test set:
```py
model_forces = mymodel.predict(tst_confs)
errors = np.sqrt(np.sum((tst_forces - model_forces)**2, axis = 1))

```

Build the mapped 2- and 3- body force field:

```py
num_2b, num_3b = 200, 100
mymodel.build_grid(grid start, num_2b, num_3b)

```
Save the model:

```py
mymodel.save("thismodel.json")

```
Setup an ASE calculator:

```py
from mff.calculators import CombinedSingleSpecies
calc = CombinedSingleSpecies(r_cut, mymodel.grid_2b, mymodel.grid_3b, rep_alpha = 1.9)

```
Assign the calculator to a previously created ASE atoms object:

```py
atoms.set_calculator(calc)

```


## Maintainers

* Claudio Zeni (claudio.zeni@kcl.ac.uk),
* Aldo Glielmo (aldo.glielmo@kcl.ac.uk),
* Ádám Fekete (adam.fekete@kcl.ac.uk).

## References

[1] A. Glielmo, C. Zeni, A. De Vita, *Efficient non-parametric n-body force fields from machine learning* (https://arxiv.org/abs/1801.04823)

[2] C .Zeni, K. Rossi, A. Glielmo, N. Gaston, F. Baletto, A. De Vita *Building machine learning force fields for nanoclusters* (https://arxiv.org/abs/1802.01417)
