import logging

import pandas as pd
import numpy as np
import wfdb
import ast

from utils.file.file import load_string_from_file
from utils.misc.datastructure import perform_shape_switch


def load_raw_ecgs_ptbxl(df, sampling_rate, path, leads_to_use=None):
    if sampling_rate == 100:
        filenames = df.filename_lr
    elif sampling_rate == 500:
        filenames = df.filename_hr
    else:
        raise Exception('Unknown sampling rate "{}"! Possible rates: [100, 500].'.format(sampling_rate))

    return load_raw_ecgs_wfdb(path, filenames, df.record_id, leads_to_use=leads_to_use)


def load_raw_ecgs_wfdb(path, filenames, record_ids, leads_to_use=None):
    filepaths = [path + f for f in filenames]

    data = {}

    for filepath, record_id in zip(filepaths, record_ids):
        logging.debug('Loading {}'.format(filepath))

        signal, meta = wfdb.rdsamp(filepath)
        signal = perform_shape_switch(signal)
        signal = np.nan_to_num(signal)

        metadata = {'sampling_rate_sec': meta['fs'],
                    'unitofmeasurement': meta['units'][0],
                    'length_sec': meta['sig_len'] / meta['fs'],
                    'length_timesteps': meta['sig_len']}

        leads = {}

        for lead_id, lead_signal in zip(meta['sig_name'], signal):
            if leads_to_use is None:
                leads[lead_id] = lead_signal

            elif lead_id in leads_to_use:
                leads[lead_id] = lead_signal

        data[record_id] = {'leads': leads, 'metadata': metadata}

    return data


def load_raw_ecgs_shareedb(path, leads_to_use=None):
    records_file = path + 'RECORDS'
    record_ids = load_string_from_file(records_file).split('\n')
    record_ids = record_ids[1:3] #TODO: tmp

    return load_raw_ecgs_wfdb(path, record_ids, record_ids, leads_to_use=leads_to_use)


def derive_diagnostic_superclass(label_dict, scp_records, threshold=50):
    collected = []

    # For each SCP code in list, derive superclass from SCP records
    for scp_code in label_dict:
        # Consider only probable SCP codes for diagnosis derivation
        if label_dict[scp_code] >= threshold:
            # Check if SCP code is known and derivation rule exists
            if scp_code in scp_records.index:
                collected.append(scp_records.loc[scp_code].diagnostic_class)

    return list(set(collected))


def derive_diagnostic_subclass(label_dict, scp_records, threshold=50):
    collected = []

    # For each SCP code in list, derive subclass from SCP records
    for scp_code in label_dict:
        # Consider only probable SCP codes for diagnosis derivation
        if label_dict[scp_code] >= threshold:
            # Check if SCP code is known and derivation rule exists
            if scp_code in scp_records.index:
                collected.append(scp_records.loc[scp_code].diagnostic_subclass)

    return list(set(collected))


def only_normal(labels):
    # Look for NORM records, exclude all that are not exclusively normal (e.g. CD, STTC)
    if 'NORM' in labels and len(labels) == 1:
        return 1
    else:
        return 0


def normal_and_others(labels):
    if 'NORM' in labels and len(labels) > 1:
        return 1
    else:
        return 0


def myocardial_infarction(labels):
    if 'MI' in labels:
        return 1
    else:
        return 0


def anterior_myocardial_infarction(labels):
    if 'AMI' in labels:
        return 1
    else:
        return 0


def inferior_myocardial_infarction(labels):
    if 'IMI' in labels:
        return 1
    else:
        return 0


def lateral_myocardial_infarction(labels):
    if 'LMI' in labels:
        return 1
    else:
        return 0


def posterior_myocardial_infarction(labels):
    if 'PMI' in labels:
        return 1
    else:
        return 0


def get_record_id_from_filename(filename):
    s = filename.replace('records100/', '')
    s = s.replace('_lr', '')
    _, s = s.split('/')

    return 'PTB-XL-{}'.format(s)


def calc_BMI(height_cm, weight_kg):
    height_m = height_cm / 100
    return round(weight_kg / (height_m) ** 2, 2)


def calc_BSA(height_cm, weight_kg):
    """Source: Mosteller formula, DOI: 10.1056/NEJM198710223171717"""
    return round(np.math.sqrt((height_cm * weight_kg) / 3600), 2)


def obesity_check(bmi):
    if bmi >= 30:
        return 1
    else:
        return 0


def load_patientrecords_mi_norm(path, record_ids_excluded=None):
    # Load annotation- and metadata into df
    db_records = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')

    # Convert SCP string to dict for each record in df
    db_records.scp_codes = db_records.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Add record_id column
    db_records['record_id'] = db_records.filename_lr.apply(lambda x: get_record_id_from_filename(x))

    # Exclude certain records if required
    if record_ids_excluded is not None:
        included_sel = db_records.record_id.apply(lambda x: check_included(x, record_ids_excluded))
        db_records = db_records[included_sel]

    # Load SCP records for diagnostic aggregation
    scp_records = pd.read_csv(path + 'scp_statements.csv', index_col=0)

    # Filter df for diagnostic SCP codes only (others: rhythm, form)
    scp_records = scp_records[scp_records.diagnostic == 1]

    # Derive diagnostic superclass and add column to df
    db_records['diagnostic_superclass'] = db_records.scp_codes.apply(lambda x: derive_diagnostic_superclass(x, scp_records))
    db_records['diagnostic_subclass'] = db_records.scp_codes.apply(lambda x: derive_diagnostic_subclass(x, scp_records))

    # Create columns for NORM and MI superclasses as well as MI subclasses
    db_records['NORM'] = db_records.diagnostic_superclass.apply(lambda x: only_normal(x))
    db_records['MI'] = db_records.diagnostic_superclass.apply(lambda x: myocardial_infarction(x))
    db_records['AMI'] = db_records.diagnostic_subclass.apply(lambda x: anterior_myocardial_infarction(x))
    db_records['IMI'] = db_records.diagnostic_subclass.apply(lambda x: inferior_myocardial_infarction(x))
    db_records['LMI'] = db_records.diagnostic_subclass.apply(lambda x: lateral_myocardial_infarction(x))
    db_records['PMI'] = db_records.diagnostic_subclass.apply(lambda x: posterior_myocardial_infarction(x))

    # Collect normal and MI records and concatenate dfs
    records_norm = db_records[db_records['NORM'] == 1]
    records_mi = db_records[db_records['MI'] == 1]
    records = pd.concat([records_mi, records_norm])

    return records


def check_included(x, record_ids_excluded):
    if x in record_ids_excluded:
        return False
    else:
        return True


def load_norm_and_mi_ecgs(path, sampling_rate, leads_to_use, record_ids_excluded=None):
    records = load_patientrecords_mi_norm(path, record_ids_excluded)
    ecgs = load_raw_ecgs_ptbxl(records, sampling_rate, path, leads_to_use)

    return ecgs
