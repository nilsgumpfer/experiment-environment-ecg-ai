"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Abstract class for loading of preprocessed data

 Copyright 2020 The Authors. All Rights Reserved.
"""

from abc import ABC, ABCMeta, abstractmethod


class AbstractDataloader(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, params, dataset_directory='../../data/datasets/', split_directory='../../data/splits/'):
        super().__init__()
        self.params = params
        self.dataset_directory = dataset_directory
        self.split_directory = split_directory

    @abstractmethod
    def load_data(self):
        pass




