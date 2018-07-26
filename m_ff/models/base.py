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


class TwoBodyModel(Model, metaclass=ABCMeta):
    pass


class ThreeBodyModel(Model, metaclass=ABCMeta):
    pass


class CombinedModel(Model, metaclass=ABCMeta):
    pass


class SingleSpeciesModel(Model, metaclass=ABCMeta):

    def __init__(self, element, r_cut):
        super().__init__()

        self.element = element
        self.r_cut = r_cut

    @property
    def parameters(self):
        params = super().parameters
        params['elements'] = [self.element]
        params['r_cut'] = self.r_cut
        return params


class TwoSpeciesModel(Model, metaclass=ABCMeta):
    def __init__(self, elements, r_cut):
        super().__init__()

        self.elements = elements
        self.r_cut = r_cut

    @property
    def parameters(self):
        params = super().parameters
        params['elements'] = self.elements
        params['r_cut'] = self.r_cut
        return params
