import logging

from preprocessing.abstract_preprocessor import AbstractPreprocessor
from utils.data.data import validate_and_clean_clinical_parameters_for_records, \
    categorize_clinical_parameters_for_records, \
    one_hot_encode_clinical_parameters_for_records, \
    combine_ecgs_and_clinical_parameters, save_dataset, load_ecgs_from_ptbxl, \
    load_clinical_parameters_from_ptbxl_snapshot, load_metadata, derive_ecg_variants_multi


class PtbxlPreprocessor(AbstractPreprocessor):

    def __init__(self, params):
        super().__init__(params)

    def perform_preprocessing(self):
        # 1. Load ECGs
        logging.info('Loading ECGs from snaphot...')
        original_ecgs = load_ecgs_from_ptbxl(self.params['snapshot_id'], leads_to_use=self.params['leads_to_use'], record_ids_excluded=self.params['record_ids_excluded'])
        logging.info('Loaded ECGs from snaphot')

        # 2. Further ECG derivation
        derived_ecgs = derive_ecg_variants_multi(original_ecgs, ['ecg_raw'])
        logging.info('Derived further ECG variants')

        # 3. Load clinical parameters
        clinical_parameters = load_clinical_parameters_from_ptbxl_snapshot(self.params['snapshot_id'], self.params['clinical_parameters_inputs'], self.params['clinical_parameters_outputs'], record_ids_excluded=self.params['record_ids_excluded'])
        logging.info('Loaded clinical parameters from snapshot')

        # 4. Load Metadata
        metadata = load_metadata(self.params['metadata_id'])

        # 5. Validity check / replace values (blanks, 99, 88, etc.)
        valid_clinical_parameters = validate_and_clean_clinical_parameters_for_records(clinical_parameters, metadata)
        logging.info('Validated and cleaned clinical parameters')

        # 6. Categorization
        categorized_clinical_parameters = categorize_clinical_parameters_for_records(valid_clinical_parameters, metadata)
        logging.info('Categorized clinical parameters')

        # 7. One-hot-encoding
        one_hot_encoded_clinical_parameters = one_hot_encode_clinical_parameters_for_records(categorized_clinical_parameters, metadata)
        logging.info('One-hot encoded clinical parameters')

        # 8. Combination with ECGs
        combined_records = combine_ecgs_and_clinical_parameters(derived_ecgs, one_hot_encoded_clinical_parameters)
        logging.info('Combined ECGs and clinical parameters')

        # 9. Save dataset as file
        save_dataset(combined_records, self.params['dataset_id'])
        logging.info('Saved dataset')
