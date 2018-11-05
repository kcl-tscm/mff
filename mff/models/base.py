# -*- coding: utf-8 -*-

from abc import ABCMeta
from pathlib import Path


class Model(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grid = dict()

    def save(self, path: Path):
        pass

    @classmethod
    def load(cls):
        pass

    @property
    def parameters(self):
        return dict()
