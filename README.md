# Machine learning derived non-parametric force fields (M-FFs)

To read the full documentation check https://m-ff.readthedocs.io/en/master/index.html

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

