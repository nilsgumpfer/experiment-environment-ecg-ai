"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Abstract class for a model

 Copyright 2020 The Authors. All Rights Reserved.
"""
import logging
from abc import ABC, ABCMeta, abstractmethod
from tensorflow.keras.models import model_from_json, model_from_yaml

from utils.experiments.model import save_model, load_model


class AbstractModel(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, params):
        super().__init__()
        self.model = None
        self.params = params

        assert self.params is not None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    def save_model(self, path, epoch):
        save_model(self.model, path, epoch)

    def load_model(self, path, epoch):
        self.model = load_model(path, epoch)






