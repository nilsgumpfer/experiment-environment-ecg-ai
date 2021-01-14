from abc import ABC, ABCMeta, abstractmethod


class AbstractEvaluator(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, params, experiment_id):
        super().__init__()
        self.params = params
        self.experiment_id = experiment_id

    @abstractmethod
    def perform_evaluation(self):
        pass




