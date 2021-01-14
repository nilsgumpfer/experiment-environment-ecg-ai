import logging
from datetime import datetime
import os

import json
import numpy as np
import pandas as pd
from os import path

from sklearn.utils import shuffle, resample
from xml.dom import minidom

from utils.api.wfdb_api import load_norm_and_mi_ecgs, load_patientrecords_mi_norm
from utils.data.validation import validate_and_clean_float, validate_and_clean_char
from utils.file.file import save_dict_as_json, save_string_to_file, load_string_from_file, load_dict_from_json, \
    pickle_data, unpickle_data, make_dirs_if_not_present
from utils.api.redcap_api import load_report, load_file
from utils.misc.datastructure import perform_shape_switch


def create_snapshot_from_redcap_api(snapshot_directory='../data/kerckhoff/snapshots', datasource='schiller'):
    report_ids = {'schiller': 253, 'cardiosoft': 252}
    variable_ids = {'schiller': 'varid_ekg_schiller_pdf', 'cardiosoft': 'varid_ekg_pdf'}

    if datasource not in report_ids.keys():
        raise Exception('Unsupported datatype "{}". Please use one of "{}" instead.'.format(datasource, list(report_ids.keys())))

    snapshot_id = datetime.now().strftime("%Y-%m-%d")

    snapshot_id = '{}_{}'.format(snapshot_id, datasource)

    snapshot_path = '{}/{}'.format(snapshot_directory, snapshot_id)
    snapshot_path_clinicalparameters = '{}/{}'.format(snapshot_path, 'clinicalparameters')
    snapshot_path_ecg = '{}/{}'.format(snapshot_path, 'ecg')

    if path.exists(snapshot_path):
        raise Exception('A snapshot with ID {} already exists! Please manually clean the directory if you want to renew the snapshot. Path: {}'.format(snapshot_id, snapshot_path))
    else:
        os.makedirs(snapshot_path)
        os.mkdir(snapshot_path_clinicalparameters)
        os.mkdir(snapshot_path_ecg)

    report = load_report(report_id=report_ids[datasource])

    for record, i in zip(report, range(len(report))):
        print('Processing record {} of {}'.format(i+1, len(report)))

        record_id = record['record_id']

        try:
            ecg_data = load_file(record_id=record_id, field=variable_ids[datasource], event='baseline_arm_1')

            if '<error>' in ecg_data:
                raise Exception(ecg_data)

            filepath = '{}/{}.csv'.format(snapshot_path_ecg, record_id)
            save_string_to_file(ecg_data, filepath)

            filepath = '{}/{}.json'.format(snapshot_path_clinicalparameters, record_id)
            save_dict_as_json(record, filepath)

        except Exception as e:
            logging.warning('Skipping record {} due to Exception: {}'.format(record_id, e))

    print('Created snapshot with name "{}" at "{}"'.format(snapshot_id, snapshot_directory))


def parse_ecg_xml(xmlcode, leads_to_use=None, sampling_rate=500):
    xmlparsed = minidom.parseString(xmlcode)
    itemlist = xmlparsed.getElementsByTagName('sequence')

    leads = {}
    uom = ''
    length = 0

    for i in range(0, 12):
        cur_sequence = itemlist[i + 1]
        lead = list(np.fromstring(cur_sequence.getElementsByTagName('digits')[0].childNodes[0].nodeValue,
                                  dtype=int,
                                  sep=' '))
        length = len(lead)
        uom = cur_sequence.getElementsByTagName('scale')[0].getAttribute('unit')
        lead_id = cur_sequence.getElementsByTagName('code')[0].getAttribute('code').replace('MDC_ECG_LEAD_', '')

        if leads_to_use is None:
            leads[lead_id] = lead

        elif lead_id in leads_to_use:
            leads[lead_id] = lead

    metadata = {'sampling_rate_sec': sampling_rate,
                'unitofmeasurement': uom,
                'length_sec': int(length / sampling_rate),
                'length_timesteps': length}

    return leads, metadata


def load_ecg_dynamic(path, leads_to_use=None):
    if path.endswith('xml'):
        return load_ecg_xml(path, leads_to_use=leads_to_use)
    elif path.endswith('csv'):
        return load_ecg_csv(path, leads_to_use=leads_to_use)
    else:
        raise Exception('Unknown file type for ECG: "{}"'.format(path))


def load_ecg_xml(path, leads_to_use=None):
    xmlcode = load_string_from_file(path)
    leads, metadata = parse_ecg_xml(xmlcode, leads_to_use)

    return leads, metadata


def load_ecg_csv(path, sampling_rate=500, uom='uV', leads_to_use=None):
    ecg = pd.read_csv(path)

    leads = {}

    for columnname in ecg.keys():
        lead_id = str(columnname).upper()

        if leads_to_use is not None:
            if lead_id not in leads_to_use:
                continue

        leads[lead_id] = np.asarray(list(ecg[columnname]))

    metadata = {'sampling_rate_sec': sampling_rate,
                'unitofmeasurement': uom,
                'length_sec': int(len(ecg) / sampling_rate),
                'length_timesteps': len(ecg)}

    return leads, metadata


def save_ecg_csv(path, ecg):
    logging.debug('Saving {}'.format(path))
    leads = ecg['leads']
    df = pd.DataFrame.from_dict(leads)
    df.to_csv(path)


def load_ecgs_from_redcap_snapshot(snapshot, leads_to_use, record_ids_excluded,
                                   snapshot_directory='../../data/kerckhoff/snapshots'):
    path = snapshot_directory + '/{}/ecg/'.format(snapshot)
    ecgfiles = os.listdir(path)

    ecgs = {}

    for filename in ecgfiles:
        exclude = False
        record_id = filename.replace('.xml', '')

        if record_ids_excluded is not None:
            if record_id in record_ids_excluded:
                exclude = True
                logging.info('Excluded record "{}" from dataloading (ECG)'.format(record_id))

        if exclude is False:
            leads, metadata = load_ecg_xml(path + filename, leads_to_use)
            ecgs[record_id] = {'leads': leads, 'metadata': metadata}

    return ecgs


def load_ecgs_from_redcap_snapshots(snapshots, leads_to_use, record_ids_excluded,
                                    snapshot_directory='../../data/kerckhoff/snapshots'):
    ecgs = {}

    for snapshot in snapshots:
        path = snapshot_directory + '/{}/ecg/'.format(snapshot)
        ecgfiles = os.listdir(path)

        for filename in ecgfiles:
            exclude = False

            if filename.endswith('xml'):
                record_id = filename.replace('.xml', '')
            elif filename.endswith('csv'):
                record_id = filename.replace('.csv', '')
            else:
                raise Exception('Unknown file type for ECG: "{}"'.format(filename))

            if record_ids_excluded is not None:
                if record_id in record_ids_excluded:
                    exclude = True
                    logging.info('Excluded record "{}" from dataloading (ECG)'.format(record_id))

            if exclude is False:
                if record_id in ecgs.keys():
                    raise Exception('ECGs for record-id "{}" contained in multiple snapshots. Aborting.'.format(record_id))

                leads, metadata = load_ecg_dynamic(path + filename, leads_to_use)
                ecgs[record_id] = {'leads': leads, 'metadata': metadata}

    return ecgs


def load_ecgs_from_ptbxl(snapshots, sampling_rate=500, leads_to_use=None,
                         snapshot_directory='../../data/ptbxl/snapshots', record_ids_excluded=None):
    path = snapshot_directory + '/{}/'.format(snapshots[0])
    ecgs = load_norm_and_mi_ecgs(path, sampling_rate, leads_to_use, record_ids_excluded)

    return ecgs


def load_clinical_parameters_json(path, params_input, params_output):
    allparams = load_dict_from_json(path)

    return extract_clinical_parameters_from_df(allparams, params_input, params_output)


def extract_clinical_parameters_from_df(df, params_input, params_output):
    inputs = {}
    outputs = {}

    if params_input is not None:
        for param in params_input:
            try:
                inputs[param] = df[param]
            except KeyError:
                raise Exception('Unknown clinical input parameter "{}". Aborting.'.format(param))

        assert (len(inputs)) > 0

    for param in params_output:
        try:
            outputs[param] = df[param]
        except KeyError:
            raise Exception('Unknown clinical output parameter "{}". Aborting.'.format(param))

    assert (len(outputs)) > 0

    return inputs, outputs


def load_metadata(metadata_id, metadata_directory='./../../data/metadata/'):
    path = metadata_directory + metadata_id + '.json'

    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata

    except FileNotFoundError:
        raise Exception('Metadata file at "{}" does not exist. Aborting.'.format(path))

    except json.decoder.JSONDecodeError as e:
        raise Exception(
            'Metadata file at "{}" contains errors. JSON could not be parsed. Aborting. Error message: {}'.format(path,
                                                                                                                  str(
                                                                                                                      e)))


def one_hot_encode_clinical_parameters(clinical_parameters, metadata):
    encoded = {}

    for param in clinical_parameters:
        value = clinical_parameters[param]
        try:
            encoded[param] = np.array(metadata[param]['values_one_hot'][value])
        except KeyError:
            raise Exception(
                'One hot encoding failed because of missing rule for clinical parameter "{}" and value "{}". Check value or implement rule!'.format(
                    param, value))

    return encoded


def scale_ecg(ecg, factor):
    for lead_id in ecg['leads']:
        lead = np.array(ecg['leads'][lead_id])
        ecg['leads'][lead_id] = lead * factor

    if factor == 1 / 1000 and ecg['metadata']['unitofmeasurement'] == 'uV':
        ecg['metadata']['unitofmeasurement'] = 'mV'
    else:
        ecg['metadata']['unitofmeasurement'] = ecg['metadata']['unitofmeasurement'] + '*' + str(factor)

    return ecg


def scale_ecgs(ecgs, factor):
    scaled_ecgs = {}

    for record_id in ecgs:
        scaled_ecgs[record_id] = scale_ecg(ecgs[record_id], factor)

    return scaled_ecgs


def derive_ecg_variants_multi(ecgs, variants):
    derived_ecgs = {}

    for record_id in ecgs:
        derived_ecgs[record_id] = derive_ecg_variants(ecgs[record_id], variants)

    return derived_ecgs


def calculate_delta_for_lead(lead):
    delta_list = []

    for index in range(0, len(lead) - 1):
        delta_list.append(lead[index + 1] - lead[index])

    delta_list = np.round(np.array(delta_list), 6)

    return delta_list


def calculate_delta_for_leads(leads):
    delta_leads = {}

    for lead_id in leads:
        delta_leads[lead_id] = calculate_delta_for_lead(leads[lead_id])

    return delta_leads


def derive_ecg_from_delta_for_lead(delta_lead):
    value_list = []
    v = delta_lead[0]

    for index in range(0, len(delta_lead)):
        v += delta_lead[index]
        value_list.append(v)

    value_list = np.round(np.array(value_list), 6)

    return value_list


def derive_ecg_from_delta_for_leads(delta_leads):
    leads = {}

    for lead_id in delta_leads:
        leads[lead_id] = derive_ecg_from_delta_for_lead(delta_leads[lead_id])

    return leads


def derive_ecg_variants(ecg, variants):
    derived_ecg = {}
    for variant in variants:
        if variant == 'ecg_raw':
            derived_ecg[variant] = ecg['leads']
        elif variant == 'ecg_delta':
            derived_ecg[variant] = calculate_delta_for_leads(ecg['leads'])

    derived_ecg['metadata'] = ecg['metadata']

    return derived_ecg


def crop_ecg(ecg, start, end):
    for lead_id in ecg['leads']:
        lead = np.array(ecg['leads'][lead_id])
        ecg['leads'][lead_id] = lead[start:end]

    ecg['metadata']['crop_start'] = start
    ecg['metadata']['crop_end'] = end

    update_metadata_length(ecg, start, end)

    return ecg


def load_crops(crop_id, crop_path='../../data/crops/'):
    return load_dict_from_json(crop_path + crop_id + '.json')


def save_crops(crops, crop_id, crop_path='../../data/crops/'):
    return save_dict_as_json(crops, crop_path + crop_id + '.json')


def crop_ecgs(ecgs, crop_id):
    if crop_id is not None:
        crops = load_crops(crop_id)
        cropped_ecgs = {}

        for record_id in ecgs:
            try:
                start = crops[record_id]['start']
                end = crops[record_id]['end']
                cropped_ecgs[record_id] = crop_ecg(ecgs[record_id], start, end)
            except KeyError:
                logging.warning(
                    'No crop markers found for record {}. Please check if this was intended. Proceeding anyway.'.format(
                        record_id))
                cropped_ecgs[record_id] = ecgs[record_id]

        return cropped_ecgs

    else:
        return ecgs


def extract_subsample_from_leads_dict_based(leads, start, end):
    leads_subsampled = {}

    for lead_id in leads:
        leads_subsampled[lead_id] = leads[lead_id][start:end]

    return leads_subsampled


def update_metadata_length(ecg, start, end):
    secs_old = ecg['metadata']['length_sec']
    timesteps_old = ecg['metadata']['length_timesteps']

    timesteps_new = end - start
    secs_new = round(timesteps_new * secs_old / timesteps_old, 1)

    ecg['metadata']['length_sec'] = secs_new
    ecg['metadata']['length_timesteps'] = timesteps_new


def update_length_in_metadata(metadata, start, end):
    secs_old = metadata['length_sec']
    timesteps_old = metadata['length_timesteps']

    timesteps_new = end - start
    secs_new = round(timesteps_new * secs_old / timesteps_old, 1)

    metadata['length_sec'] = secs_new
    metadata['length_timesteps'] = timesteps_new


def extract_subsample_from_ecg_dict_based(ecg, start, end):
    subsample_ecg = {}

    for elem in ecg:
        if str(elem).startswith('ecg_'):
            subsample_ecg[elem] = extract_subsample_from_leads_dict_based(ecg[elem], start, end)
        else:
            subsample_ecg[elem] = dict(ecg[elem])

    subsample_ecg['metadata']['subsample_start'] = start
    subsample_ecg['metadata']['subsample_end'] = end

    update_metadata_length(subsample_ecg, start, end)

    return subsample_ecg


def extract_subsample_from_ecg_matrix_based(ecg, start, end):
    return ecg[start:end]


def subsample_ecgs(ecgs, subsampling_factor, window_size, ecg_variant='ecg_raw'):
    collected_subsamples = []  # TODO vielleicht dict für untersch. ekg varianten
    collected_diagnoses = []
    collected_clinical_parameters = []
    collected_metadata = []
    collected_record_ids = []

    for record_id in ecgs:
        start = 0
        record = ecgs[record_id]
        metadata = record['metadata']
        length = metadata['length_timesteps']
        ecg = convert_lead_dict_to_matrix(record[ecg_variant])
        clinical_parameters = concatenate_one_hot_encoded_parameters(record['clinical_parameters_inputs'])
        diagnosis = concatenate_one_hot_encoded_parameters(record['clinical_parameters_outputs'])

        if not length > window_size:
            raise Exception(
                'Record "{}" is not longer ({}) than the configured subsampling window size of {} timesteps. Aborting.'.format(
                    record_id, length, window_size))

        stride = int((length - window_size) / (subsampling_factor - 1))

        for i in range(subsampling_factor):
            end = start + window_size

            if end > length:
                break

            subsample = extract_subsample_from_ecg_matrix_based(ecg, start, end)

            record_id_new = '{}_{}'.format(record_id, i)
            metadata_new = dict(metadata)
            update_length_in_metadata(metadata_new, start, end)
            metadata_new['subsample_start'] = start
            metadata_new['subsample_end'] = end
            metadata_new['original_record_id'] = record_id
            metadata_new['record_id'] = record_id_new

            collected_subsamples.append(subsample)
            collected_clinical_parameters.append(clinical_parameters)
            collected_diagnoses.append(diagnosis)
            collected_metadata.append(metadata_new)
            collected_record_ids.append(record_id_new)

            start = start + stride

    return collected_record_ids, collected_metadata, collected_diagnoses, collected_clinical_parameters, collected_subsamples


def load_clinical_parameters_from_redcap_snapshots(snapshots, clinical_parameters_inputs, clinical_parameters_outputs,
                                                   record_ids_excluded,
                                                   snapshot_directory='../../data/kerckhoff/snapshots'):
    clinicalparameters = {}

    for snapshot in snapshots:
        path = snapshot_directory + '/{}/clinicalparameters/'.format(snapshot)
        parameterfiles = os.listdir(path)

        for filename in parameterfiles:
            exclude = False
            record_id = filename.replace('.json', '')

            if record_ids_excluded is not None:
                if record_id in record_ids_excluded:
                    exclude = True
                    logging.info('Excluded record "{}" from dataloading (clinical parameters)'.format(record_id))

            if exclude is False:
                if record_id in clinicalparameters.keys():
                    raise Exception('Clinical parameters for record-id "{}" contained in multiple snapshots. Aborting.'.format(record_id))

                inputs, outputs = load_clinical_parameters_json(path + filename, clinical_parameters_inputs,
                                                                clinical_parameters_outputs)
                clinicalparameters[record_id] = {'clinical_parameters_inputs': inputs,
                                                 'clinical_parameters_outputs': outputs}

    return clinicalparameters


def load_clinical_parameters_from_redcap_snapshot(snapshot, clinical_parameters_inputs, clinical_parameters_outputs,
                                                  record_ids_excluded,
                                                  snapshot_directory='../../data/kerckhoff/snapshots'):
    path = snapshot_directory + '/{}/clinicalparameters/'.format(snapshot)
    parameterfiles = os.listdir(path)

    clinicalparameters = {}

    for filename in parameterfiles:
        exclude = False
        record_id = filename.replace('.json', '')

        if record_ids_excluded is not None:
            if record_id in record_ids_excluded:
                exclude = True
                logging.info('Excluded record "{}" from dataloading (clinical parameters)'.format(record_id))

        if exclude is False:
            inputs, outputs = load_clinical_parameters_json(path + filename, clinical_parameters_inputs,
                                                            clinical_parameters_outputs)
            clinicalparameters[record_id] = {'clinical_parameters_inputs': inputs,
                                             'clinical_parameters_outputs': outputs}

    return clinicalparameters


def add_aggregated_segment_information(record):
    segmentation = {1: ['varid_1659', 'varid_1677', 'varid_1695', 'varid_1713'],
                    2: ['varid_1660', 'varid_1678', 'varid_1696', 'varid_1714'],
                    3: ['varid_1661', 'varid_1679', 'varid_1697', 'varid_1715'],
                    4: ['varid_1662', 'varid_1680', 'varid_1698', 'varid_1716'],
                    5: ['varid_1663', 'varid_1681', 'varid_1699', 'varid_1717'],
                    6: ['varid_1664', 'varid_1682', 'varid_1700', 'varid_1718'],
                    7: ['varid_1665', 'varid_1683', 'varid_1701', 'varid_1719'],
                    8: ['varid_1666', 'varid_1684', 'varid_1702', 'varid_1720'],
                    9: ['varid_1667', 'varid_1685', 'varid_1703', 'varid_1721'],
                    10: ['varid_1668', 'varid_1686', 'varid_1704', 'varid_1722'],
                    11: ['varid_1669', 'varid_1687', 'varid_1705', 'varid_1723'],
                    12: ['varid_1670', 'varid_1688', 'varid_1706', 'varid_1724'],
                    13: ['varid_1671', 'varid_1689', 'varid_1707', 'varid_1725'],
                    14: ['varid_1672', 'varid_1690', 'varid_1708', 'varid_1726'],
                    15: ['varid_1673', 'varid_1691', 'varid_1709', 'varid_1727'],
                    16: ['varid_1674', 'varid_1692', 'varid_1710', 'varid_1728'],
                    17: ['varid_1675', 'varid_1693', 'varid_1711', 'varid_1729']}

    regionmap = {'LE_anterior': [1, 7],
                 'LE_anteroseptal': [2, 8],
                 'LE_inferoseptal': [3, 9],
                 'LE_inferior': [4, 10],
                 'LE_inferolateral': [5, 11],
                 'LE_anterolateral': [6, 12],
                 'LE_apical': [13, 14, 15, 16, 17]}

    aggregation = {}

    for segment in segmentation:
        k = 'LE_segment_{}'.format(segment)
        aggregation[k] = 0

        for column in segmentation[segment]:
            if record[column] == '1':
                aggregation[k] = 1

    for region in regionmap:
        aggregation[region] = 0

        for segment in regionmap[region]:
            if aggregation['LE_segment_{}'.format(segment)] == 1:
                aggregation[region] = 1

    record.update(aggregation)

    return record


def add_norm_information(record):
    if record['varid_1657'] == '1':
        record['NO_LE'] = 0
        record['LE'] = 1
    else:
        record['NO_LE'] = 1
        record['LE'] = 0

    return record


def load_patientrecords_scar(snapshot, snapshot_directory='../data/kerckhoff/snapshots'):
    path = snapshot_directory + '/{}/clinicalparameters/'.format(snapshot)
    parameterfiles = os.listdir(path)

    records = []

    for filename in parameterfiles:
        record = load_dict_from_json(path + filename)
        record = add_aggregated_segment_information(record)
        record = add_norm_information(record)
        records.append(record)

    df = pd.DataFrame(records)

    return df


def load_clinical_parameters_from_ptbxl_snapshot(snapshot, clinical_parameters_inputs, clinical_parameters_outputs,
                                                 snapshot_directory='../../data/ptbxl/snapshots',
                                                 record_ids_excluded=None):
    path = snapshot_directory + '/{}/'.format(snapshot)
    records = load_patientrecords_mi_norm(path, record_ids_excluded)
    clinicalparameters = {}

    for index, row in records.iterrows():
        inputs, outputs = extract_clinical_parameters_from_df(row, clinical_parameters_inputs,
                                                              clinical_parameters_outputs)
        clinicalparameters[row.record_id] = {'clinical_parameters_inputs': inputs,
                                             'clinical_parameters_outputs': outputs}

    return clinicalparameters


def validate_and_clean_clinical_parameters_for_records(records, metadata):
    validated_and_cleaned = {}

    for recid in records:
        try:
            try:  # In case of unexpected values for outputs, skip current record
                outputs = validate_and_clean_clinical_parameters(records[recid]['clinical_parameters_outputs'],
                                                                 metadata)
            except ValueError:
                logging.warning(
                    'Unexpected value in clinical parameters (outputs) for record "{}". Please check values: "{}". Skipping record.'.format(
                        recid, records[recid]['clinical_parameters_outputs']))
                continue

            inputs = validate_and_clean_clinical_parameters(records[recid]['clinical_parameters_inputs'], metadata)
        except Exception as e:  # In case of other exceptions, raise new exception with record-id information added
            raise Exception('Record-ID {}: {}'.format(recid, e))

        validated_and_cleaned[recid] = {'clinical_parameters_inputs': inputs, 'clinical_parameters_outputs': outputs}

    return validated_and_cleaned


def validate_and_clean_clinical_parameters(clinical_parameters, metadata):
    validated_and_cleaned = {}

    for param in clinical_parameters:
        value = clinical_parameters[param]

        if metadata[param]['type'] == 'char':
            value_vc = validate_and_clean_char(param, str(value),
                                               metadata[param]['values_allowed'],
                                               metadata[param]['values_replace'])
        elif metadata[param]['type'] == 'float':
            value_vc = validate_and_clean_float(param, value,
                                                metadata[param]['valmin'],
                                                metadata[param]['valmax'])
        else:
            raise Exception('Unkown parameter: "{}". Please implement validation and cleansing rule!'.format(param))

        validated_and_cleaned[param] = value_vc

    return validated_and_cleaned


def categorize_clinical_parameters(clinical_parameters, metadata):
    for param in clinical_parameters:

        if metadata[param]['type'] == 'float':
            rules = metadata[param]['categorization_rules']

            for category in rules:
                if category['start'] <= clinical_parameters[param] < category['end']:
                    clinical_parameters[param] = category['name']
                    break

    return clinical_parameters


def categorize_clinical_parameters_for_records(records, metadata):
    categorized = {}

    for recid in records:
        inputs = categorize_clinical_parameters(records[recid]['clinical_parameters_inputs'], metadata)
        outputs = categorize_clinical_parameters(records[recid]['clinical_parameters_outputs'], metadata)
        categorized[recid] = {'clinical_parameters_inputs': inputs, 'clinical_parameters_outputs': outputs}

    return categorized


def one_hot_encode_clinical_parameters_for_records(records, metadata):
    onehot_encoded = {}

    for recid in records:
        inputs = one_hot_encode_clinical_parameters(records[recid]['clinical_parameters_inputs'], metadata)
        outputs = one_hot_encode_clinical_parameters(records[recid]['clinical_parameters_outputs'], metadata)
        onehot_encoded[recid] = {'clinical_parameters_inputs': inputs, 'clinical_parameters_outputs': outputs}

    return onehot_encoded


def combine_ecgs_and_clinical_parameters(ecgs, clinical_parameters):
    combined = {}

    for record_id in ecgs:
        ecg = ecgs[record_id]

        try:
            cp = clinical_parameters[record_id]
        except KeyError:
            logging.warning(
                'No clinical parameters available in datapipeline for record "{}". Skipping record.'.format(record_id))
            continue

        combined[record_id] = dict(ecg)
        combined[record_id].update(cp)

    return combined


def save_dataset(records, dataset_id, dataset_directory='../../data/datasets/'):
    make_dirs_if_not_present(dataset_directory)
    pickle_data(records, dataset_directory + dataset_id + '.pickled')


def load_dataset(dataset_id, dataset_directory='../../data/datasets/'):
    return unpickle_data(dataset_directory + dataset_id + '.pickled')


def load_split(split_id, split_directory='../../data/splits/'):
    return load_dict_from_json(split_directory + split_id + '.json')


def load_and_split_dataset(dataset_id, split_id, dataset_directory='../../data/datasets/', split_directory='../../data/splits/'):
    records_split = {}

    records = load_dataset(dataset_id, dataset_directory=dataset_directory)
    split = load_split(split_id, split_directory=split_directory)

    for group in split:
        records_split[group] = {}
        record_counter = {recid: 0 for recid in split[group]}

        for record_id in split[group]:
            try:
                if record_counter[record_id] > 0:
                    records_split[group]['{}_{}'.format(record_id, record_counter[record_id])] = records[record_id]
                else:
                    records_split[group][record_id] = records[record_id]
            except KeyError:
                raise Exception(
                    'Record "{}" contained in split group "{}" of split "{}" not available in dataset "{}". Aborting.'.format(
                        record_id, group, split_id, dataset_id))

            record_counter[record_id] += 1

    return records_split


def extract_subdict_from_dict(dct, subdict):
    collected = []

    for lv1 in dct:
        collected.append(dct[lv1][subdict])

    return collected


def concatenate_one_hot_encoded_parameters(dct):
    collected = []

    for p in dct:
        collected += list(dct[p])

    return np.array(collected)


def concatenate_one_hot_encoded_parameters_for_records(records):
    collected = []

    for record in records:
        collected.append(concatenate_one_hot_encoded_parameters(record))

    return collected


def convert_matrix_to_lead_dict_ecg(matrix_ecg, metadata):
    leads = perform_shape_switch(matrix_ecg)
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_dict = {}

    for i in range(len(lead_names)):
        x = lead_names[i]
        lead_dict[x] = leads[i]

    return {'leads': lead_dict, 'metadata': metadata}


def convert_lead_dict_to_matrix(leads, shape_switch=True):
    collected = []

    for lead_id in leads:
        collected.append(leads[lead_id])

    collected = np.asarray(collected)

    if shape_switch:
        collected = perform_shape_switch(collected)

    return collected


def convert_lead_dict_to_matrix_for_records(records):  # TODO: Performance issues -> generate matrix before subsampling
    collected = []
    first = convert_lead_dict_to_matrix(records[0])
    test = np.zeros((len(records), np.shape(first)[0], np.shape(first)[1]))

    i = 0
    for record in records:
        if i % 100 == 0:
            print(i, '/', len(records))
        # collected.append(convert_lead_dict_to_matrix(record))
        test[i] = convert_lead_dict_to_matrix(record)
        i = i + 1

    # return collected
    return test


def derive_binary_one_hot_classes_for_list_of_labels(labels):
    collected = []

    for label in labels:
        for b in ['TRUE', 'FALSE']:
            collected.append('{}_{}'.format(label, b))

    return collected


def extract_element_from_dicts(dicts, element):
    collected = []

    for d in dicts:
        collected.append(d[element])

    return collected


def save_split(split_id, records_train, records_val, split_dir='../../data/splits'):
    make_dirs_if_not_present(split_dir)
    split = {'training': records_train, 'validation': records_val}
    path = '{}/{}.json'.format(split_dir, split_id)

    if os.path.exists(path):
        raise Exception('Split file at "{}" already exists. Splits cannot be overwritten due to repoducibility guidelines. Make sure they are not used by any other experiments! Please delete them manually if you want to replace them.'.format(path))

    save_dict_as_json(split, path)


def save_test_split(split_id, records_test, split_dir='../../data/splits'):
    make_dirs_if_not_present(split_dir)
    split = {'test': records_test}
    path = '{}/{}_test.json'.format(split_dir, split_id)

    if os.path.exists(path):
        raise Exception('Split file at "{}" already exists. Splits cannot be overwritten due to repoducibility guidelines. Make sure they are not used by any other experiments! Please delete them manually if you want to replace them.'.format(path))

    save_dict_as_json(split, path)


def shuffle_based_on_random_seed(records, random_seed):
    r_sorted = sorted(records)
    r_shuffled = shuffle(r_sorted, random_state=random_seed)

    return r_shuffled


def assign_stratified_records_to_k_groups(stratification_groups, k, random_seed):
    grouped_records = {i: [] for i in range(k)}
    g = 0

    # Assign records to k groups, homogeneously based on stratification
    for sg in stratification_groups:

        # Shuffle records before assignment
        records = shuffle_based_on_random_seed(stratification_groups[sg], random_seed)

        # Assign records to groups, run through groups 0 to k-1
        for r in records:
            grouped_records[g].append(r)
            g += 1
            if g == k:
                g = 0

    return grouped_records


def generate_cross_validation_splits(split_id, stratification_groups, variables, random_seed,
                                     split_dir='../../data/splits'):
    k = variables['k']

    # Assign records to k groups, homogeneously based on stratification
    grouped_records = assign_stratified_records_to_k_groups(stratification_groups, k, random_seed)

    # k times, split records into 1 validation part and k-1 training parts
    for g in grouped_records:

        # The current g is used for validation
        records_validation = grouped_records[g]
        records_train = []

        # Use all but current group for training
        for x in grouped_records:
            if x != g:
                records_train += grouped_records[x]

        # Save split
        sub_split_id = '{}_k{}'.format(split_id, g)
        save_split(sub_split_id, records_train, records_validation, split_dir=split_dir)


def generate_bootstrapping_splits(split_id, stratification_groups, variables, random_seed,
                                  split_dir='../../data/splits'):
    n = variables['n']

    for i in range(n):
        train = []
        val = []

        seed = random_seed + i

        for sg in stratification_groups:
            # Shuffle records before assignment
            records = shuffle_based_on_random_seed(stratification_groups[sg], seed)

            # Randomly draw n=len(records) records witn replacement from all records, use for training
            train_tmp = resample(records, replace=True, n_samples=len(records), random_state=seed)

            # Use undrawn records for validation
            val_tmp = list(set(records) - set(train_tmp))

            train += train_tmp
            val += val_tmp

        # Save split
        sub_split_id = '{}_n{}'.format(split_id, i)
        save_split(sub_split_id, train, val, split_dir=split_dir)


def generate_ratio_based_split(split_id, stratification_groups, variables, random_seed, split_dir='../../data/splits'):
    train = []
    val = []
    ratio = variables['ratio_split']

    # Assign records to k groups, homogeneously based on stratification
    for sg in stratification_groups:
        # Shuffle records before assignment
        records = shuffle_based_on_random_seed(stratification_groups[sg], random_seed)
        split_point = int(len(records) * ratio)
        train_tmp = records[:split_point]
        val_tmp = records[split_point:]
        train += train_tmp
        val += val_tmp

    # Save split
    save_split(split_id, train, val, split_dir=split_dir)


def generate_splits_for_stratification_groups_and_validation_type(split_id, stratification_groups, validation_type,
                                                                  variables, random_seed,
                                                                  split_dir='../../data/splits'):
    if validation_type == 'cross_validation':
        generate_cross_validation_splits(split_id, stratification_groups, variables, random_seed, split_dir=split_dir)
    elif validation_type == 'bootstrapping':
        generate_bootstrapping_splits(split_id, stratification_groups, variables, random_seed, split_dir=split_dir)
    elif validation_type == 'single' or validation_type is None:
        generate_ratio_based_split(split_id, stratification_groups, variables, random_seed, split_dir=split_dir)
    else:
        raise Exception('Invalid validation_type! ({})'.format(validation_type))


def generate_splits_for_dataset_and_validation_type(split_id, dataset_id, validation_type, stratification_variable,
                                                    random_seed, variables, dataset_directory='../../data/datasets/',
                                                    split_directory='../../data/splits'):
    # Load records from dataset
    records = load_dataset(dataset_id, dataset_directory)
    record_ids = [r for r in records]

    # Extract label values for stratification
    values = [np.argmax(records[r]['clinical_parameters_outputs'][stratification_variable]) for r in records]

    # Create stratification groups
    stratification_groups = {v: [] for v in values}

    # Assign records to stratification groups based on strat. variable
    for r, v in zip(record_ids, values):
        stratification_groups[v].append(r)

    # In case of test data hold-out, separate test records in advance and save test split
    if 'ratio_test' in variables.keys():
        records_test = []

        for sg in stratification_groups:
            test_tmp = draw_test_records_from_population(stratification_groups[sg], variables['ratio_test'], random_seed)
            records_test += test_tmp
            stratification_groups[sg] = list(set(stratification_groups[sg]) - set(test_tmp))

        records_test = shuffle_based_on_random_seed(records_test, random_seed)
        save_test_split(split_id, records_test, split_dir=split_directory)

    # Generate splits based on stratification groups and validation type
    generate_splits_for_stratification_groups_and_validation_type(split_id, stratification_groups, validation_type,
                                                                  variables, random_seed, split_dir=split_directory)


def filter_TP_TN(array, classifications):
    ret = []

    for a, c in zip(array, classifications):
        if c in ['TP', 'TN']:
            ret.append(a)

    return ret


def draw_test_records_from_population(records, ratio, random_seed):
    records_shuffled = shuffle_based_on_random_seed(records, random_seed=random_seed)

    splitpoint = int(len(records_shuffled) * ratio)

    return records_shuffled[:splitpoint]
