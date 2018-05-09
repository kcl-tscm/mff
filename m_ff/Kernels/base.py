from abc import ABCMeta, abstractmethod


class Kernel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, kernel_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_name = kernel_name
