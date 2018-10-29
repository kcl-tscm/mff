# Machine learning derived non-parametric force fields (MFFs)

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
For a detailed description of the theory behind Gaussian process regression to predict forces and/or energies, and an explanation of the "mapping" technique used, please refer to [1].

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

A simple example on how to use the package, with typical parameters.

## Maintainers

* Claudio Zeni (claudio.zeni@kcl.ac.uk),
* Aldo Glielmo (aldo.glielmo@kcl.ac.uk),
* Ádám Fekete (adam.fekete@kcl.ac.uk).

## References

[1] A. Glielmo, C. Zeni, A. De Vita, *Efficient non-parametric n-body force fields from machine learning* (https://arxiv.org/abs/1801.04823)

[2] C .Zeni, K. Rossi, A. Glielmo, N. Gaston, F. Baletto, A. De Vita *Building machine learning force fields for nanoclusters* (https://arxiv.org/abs/1802.01417)
