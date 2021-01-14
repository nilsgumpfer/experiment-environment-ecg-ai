"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Abstract class for the preprocessing

 Copyright 2020 The Authors. All Rights Reserved.
"""

from abc import ABC, ABCMeta, abstractmethod


class AbstractPreprocessor(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, params):
        super().__init__()
        self.params = params

    @abstractmethod
    def perform_preprocessing(self):
        pass




