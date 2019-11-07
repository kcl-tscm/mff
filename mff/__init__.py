import os

from .gp import GaussianProcess

Mffpath = __path__[0] + "/cache/"

if not os.path.exists(Mffpath):
    os.mkdir(Mffpath)

__all__ = [GaussianProcess]
