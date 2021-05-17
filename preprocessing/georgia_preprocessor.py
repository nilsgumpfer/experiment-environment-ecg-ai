import logging

from preprocessing.abstract_preprocessor import AbstractPreprocessor
from utils.data.data import validate_and_clean_clinical_parameters_for_records, \
    categorize_clinical_parameters_for_records, \
    one_hot_encode_clinical_parameters_for_records, \
    combine_ecgs_and_clinical_parameters, save_dataset, load_ecgs_from_ptbxl, \
    load_clinical_parameters_from_ptbxl_snapshot, load_metadata, derive_ecg_variants_multi, load_ecgs_from_georgia


class GeorgiaPreprocessor(AbstractPreprocessor):

    def __init__(self, params):
        super().__init__(params)

    def perform_preprocessing(self):
        # 1. Load ECGs
        logging.info('Loading ECGs from snaphot...')
        original_ecgs = load_ecgs_from_georgia(self.params['snapshot_id'], leads_to_use=self.params['leads_to_use'], labels_to_use=self.params['clinical_parameters_outputs'], record_ids_excluded=self.params['record_ids_excluded'])
        logging.info('Loaded ECGs from snaphot')

        # 2. Further ECG derivation
        derived_ecgs = derive_ecg_variants_multi(original_ecgs, ['ecg_raw'])
        logging.info('Derived further ECG variants')

        # 4. Load Metadata
        metadata = load_metadata(self.params['metadata_id'])

        # 7. One-hot-encoding
        one_hot_encoded_clinical_parameters = one_hot_encode_clinical_parameters_for_records(derived_ecgs, metadata)
        logging.info('One-hot encoded clinical parameters')

        # 8. Combination with ECGs
        combined_records = combine_ecgs_and_clinical_parameters(derived_ecgs, one_hot_encoded_clinical_parameters)
        logging.info('Combined ECGs and clinical parameters')

        # 9. Save dataset as file
        save_dataset(combined_records, self.params['dataset_id'])
        logging.info('Saved dataset')
