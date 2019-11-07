import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

path = Path(os.path.abspath(__file__))
Mffpath = path.parent.parent / "cache/"

class Kernel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, kernel_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_name = kernel_name
