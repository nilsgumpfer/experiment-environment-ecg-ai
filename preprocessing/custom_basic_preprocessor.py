import logging

from preprocessing.abstract_preprocessor import AbstractPreprocessor
from utils.data.data import scale_ecgs, derive_ecg_variants_multi, \
    validate_and_clean_clinical_parameters_for_records, \
    categorize_clinical_parameters_for_records, \
    one_hot_encode_clinical_parameters_for_records, \
    combine_ecgs_and_clinical_parameters, save_dataset, load_metadata, load_ecgs_from_custom_snapshots, \
    load_clinical_parameters_from_custom_snapshots


class CustomBasicPreprocessor(AbstractPreprocessor):

    def __init__(self, params):
        super().__init__(params)

    def perform_preprocessing(self):
        # 1. Load ECGs
        original_ecgs = load_ecgs_from_custom_snapshots(self.params['snapshot_id'], self.params['leads_to_use'], self.params['record_ids_excluded'])
        logging.info('Loaded ECGs from snaphot')

        # 2. Scale ECGs
        scaleded_ecgs = scale_ecgs(original_ecgs, 1 / 1000)
        logging.info('Scaled ECGs')

        # 4. Further ECG derivation
        derived_ecgs = derive_ecg_variants_multi(scaleded_ecgs, ['ecg_raw'])
        logging.info('Derived further ECG variants')

        # 5. Load clinical parameters
        clinical_parameters = load_clinical_parameters_from_custom_snapshots(self.params['snapshot_id'], self.params['clinical_parameters_inputs'], self.params['clinical_parameters_outputs'], self.params['record_ids_excluded'])
        logging.info('Loaded clinical parameters from snapshot')

        # 6. Load Metadata
        metadata = load_metadata(self.params['metadata_id'])

        # 7. Validity check / replace values (blanks, 99, 88, etc.)
        valid_clinical_parameters = validate_and_clean_clinical_parameters_for_records(clinical_parameters, metadata)
        logging.info('Validated and cleaned clinical parameters')

        # 8. Categorization
        categorized_clinical_parameters = categorize_clinical_parameters_for_records(valid_clinical_parameters, metadata)
        logging.info('Categorized clinical parameters')

        # 9. One-hot-encoding
        one_hot_encoded_clinical_parameters = one_hot_encode_clinical_parameters_for_records(categorized_clinical_parameters, metadata)
        logging.info('One-hot encoded clinical parameters')

        # 10. Combination with ECGs
        combined_records = combine_ecgs_and_clinical_parameters(derived_ecgs, one_hot_encoded_clinical_parameters)
        logging.info('Combined ECGs and clinical parameters')

        # 11. Save dataset as file
        save_dataset(combined_records, self.params['dataset_id'])
        logging.info('Saved dataset')
