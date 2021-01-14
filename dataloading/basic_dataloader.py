import logging

from dataloading.abstract_dataloader import AbstractDataloader
from utils.data.data import subsample_ecgs, load_and_split_dataset


class BasicDataloader(AbstractDataloader):

    def __init__(self, params, dataset_directory='../../data/datasets/', split_directory='../../data/splits/'):
        super().__init__(params, dataset_directory=dataset_directory, split_directory=split_directory)

    def load_data(self):
        records_split = load_and_split_dataset(self.params['dataset_id'], self.params['split_id'], dataset_directory=self.dataset_directory, split_directory=self.split_directory)
        logging.info('Loaded dataset {}'.format(self.params['dataset_id']))

        train_record_ids, train_metadata, train_diagnosis, train_clinical_parameters, train_ecg_raw = subsample_ecgs(records_split['training'], self.params['subsampling_factor'], self.params['subsampling_window_size'])
        logging.info('Subsampled training records ({} subsamples)'.format(len(train_record_ids)))
        val_record_ids, val_metadata, val_diagnosis, val_clinical_parameters, val_ecg_raw = subsample_ecgs(records_split['validation'], self.params['subsampling_factor'], self.params['subsampling_window_size'])
        logging.info('Subsampled validation records ({} subsamples)'.format(len(val_record_ids)))

        return train_record_ids, train_metadata, train_diagnosis, train_clinical_parameters, train_ecg_raw, val_record_ids, val_metadata, val_diagnosis, val_clinical_parameters, val_ecg_raw

    def load_validation_data(self):
        records_split = load_and_split_dataset(self.params['dataset_id'], self.params['split_id'], dataset_directory=self.dataset_directory, split_directory=self.split_directory)
        logging.info('Loaded dataset {}'.format(self.params['dataset_id']))

        val_record_ids, val_metadata, val_diagnosis, val_clinical_parameters, val_ecg_raw = subsample_ecgs(records_split['validation'], self.params['subsampling_factor'], self.params['subsampling_window_size'])
        logging.info('Subsampled validation records ({} subsamples)'.format(len(val_record_ids)))

        return val_record_ids, val_metadata, val_diagnosis, val_clinical_parameters, val_ecg_raw

    def load_test_data(self):
        records_split = load_and_split_dataset(self.params['dataset_id'], self.params['split_id'] + '_test', dataset_directory=self.dataset_directory, split_directory=self.split_directory)
        logging.info('Loaded dataset {}'.format(self.params['dataset_id']))

        test_record_ids, test_metadata, test_diagnosis, test_clinical_parameters, test_ecg_raw = subsample_ecgs(records_split['test'], self.params['subsampling_factor'], self.params['subsampling_window_size'])
        logging.info('Subsampled test records ({} subsamples)'.format(len(test_record_ids)))

        return test_record_ids, test_metadata, test_diagnosis, test_clinical_parameters, test_ecg_raw
