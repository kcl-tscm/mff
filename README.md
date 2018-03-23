# Machine learning derived non-parametric force fields (M-FFs)

Description of an M-FF.

## Table of Contents

- [Background on M-FFs](#background)
- [Install](#install)
- [Usage](#usage)
- [Examples](#examples)
- [Maintainers](#maintainers)
- [References](#references)

## Background on M-FFs

A bit of theory on M-FFs, with references to articles.

## Install

How to install the package, eventually we would like something like

```sh
$ python setup.py install
```

To install from source, uncompress the source files and, from the directory containing `setup.py`, run the following command:
    
    python setup.py install

Or, to build in place, run:
    
    python setup.py build_ext --inplace

If you build in place, you will also need to add your eqtools folder to your PYTHONPATH shell variable:
    
    export PYTHONPATH=$PYTHONPATH:/path/to/where/you/put/

## Usage

Description on how to use the package.

```py
import M_FF

mff = M_FF()

mff.build(traj.xyz)

```

## Examples

A simple example on how to use the package, with typical parameters.

## Maintainers

* Claudio Zeni (claudio.zeni@kcl.ac.uk),
* Aldo Glielmo (aldo.glielmo@kcl.ac.uk),
* Ádám Fekete (adam.fekete@kcl.ac.uk).

## References

* A. Glielmo, C. Zeni, A. De Vita, *Efficient non-parametric n-body force fields from machine learning* (https://arxiv.org/abs/1801.04823)

* C .Zeni, K. Rossi, A. Glielmo, N. Gaston, F. Baletto, A. De Vita *Building machine learning force fields for nanoclusters* (https://arxiv.org/abs/1802.01417)


## CODE FUNCTIONS

1) Read XYZ file and transform it into ASE object
Input: XYZ file
Output: ASE atoms object, total number of steps
Notes: Support for other formats rather than XYZ could be implemented in the fututre, we need to disuss the implications of this.

2) Carve configurations, forces and global energies
Input: ASE atoms object, total number of steps, total number of configurations to extract (default = 3000?)
Output: confs, forces, energies .npy files
Notes: It would be nice to mantain the kind of sequential configurations extraction that is already implemented and also to keep the ratio of type of elements in the central atoms possibly proportional to the suqare root of their relative presence (as it is already in the code).

3) Train and test the GP
Input: Force, configurations (possibily energies?), cutoff,  number of training points, number of testing points (default = 0), hyperparameters sigma (landscape length) , lambda (noise), theta (cutoff decay) , also with a default value.
Output: Trained GP and GP error on test
Notes: Need to optimize the process itself, add cutoff functions to the theano kernels, and need to allow for multi-processing

4) Build the mapped energy grid
Input: Cutoff, minimum distance (default = 1.5), number of points in the grid (default 100?), trained GP with its hyperparameters, elements.
utput: 4 (mono-element) or 7 (bi-element) remapped grids (in a single file), parameters used to build it written somewhere for reference.
Notes: Built using r1, r2, r12, we need to see if there is a way to exploit symmetry and reduce the number of computations needed.

5) Force and energy interpolator
Input: configurations, mapped potential grids
Output: forces and/or energies
Notes: Use the configuration carving of function 2 to create inputs for the force and energy interpolator.
We need to update the code on this since we are now working on r1 r2 r12 instead of r1 r2 $\theta$.
