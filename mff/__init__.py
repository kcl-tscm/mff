from .gp import GaussianProcess
import os

global MFFPATH
MFFPATH = __path__[0] + "/cache/"

if not os.path.exists(MFFPATH):
    os.mkdir(MFFPATH)

__all__ = [GaussianProcess]
