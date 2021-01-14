import logging

from splitting.abstract_splitter import AbstractSplitter
from utils.data.data import generate_splits_for_dataset_and_validation_type


class BasicSplitter(AbstractSplitter):

    def __init__(self, params):
        super().__init__(params)

    def perform_splitting(self):

        variables = {}

        if self.params['validation_type'] == 'cross_validation':
            variables['k'] = self.params['folds_cross_validation']
        elif self.params['validation_type'] == 'bootstrapping':
            variables['n'] = self.params['bootstrapping_n']

        for v in ['ratio_split', 'ratio_test']:
            if self.params[v] is not None:
                variables[v] = self.params[v]

        generate_splits_for_dataset_and_validation_type(split_id=self.params['split_id'],
                                                        dataset_id=self.params['dataset_id'],
                                                        validation_type=self.params['validation_type'],
                                                        variables=variables,
                                                        stratification_variable=self.params['stratification_variable'],
                                                        random_seed=self.params['random_seed'])

        logging.info('Splits generated!')
