import configparser
import json
import logging
import os
import pickle
import shutil

import pandas as pd
from PyPDF2 import PdfWriter, PdfReader


def save_dict_as_json(dct, path):
    with open(path, 'w') as fp:
        json.dump(dct, fp)
        fp.close()


def load_dict_from_json(path):
    string = load_string_from_file(path)
    dct = json.loads(string)

    return dct


def save_string_to_file(string, path):
    with open(path, 'w') as fp:
        fp.write(string)
        fp.close()


def load_string_from_file(path):
    with open(path, 'r') as fp:
        string = fp.read()
        fp.close()

    return string


def combine_pdfs(paths, targetpath, cleanup=False):
    pdf_writer = PdfWriter()

    for path in paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    with open(targetpath, 'wb') as fh:
        pdf_writer.write(fh)

    if cleanup:
        for path in paths:
            os.remove(path)


def pickle_data(array, path):
    with open(path, 'wb') as fp:
        pickle.dump(array, fp, pickle.HIGHEST_PROTOCOL)
        fp.close()


def unpickle_data(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def cleanup_directory(path, make=False):
    shutil.rmtree(path, ignore_errors=True)
    logging.debug('Removed directory "{}"'.format(path))

    if make:
        make_dirs_if_not_present(path)


def make_dirs_if_not_present(path):
    os.makedirs(path, exist_ok=True)


def logdir_exists(experiment_id, category):
    logdir = '../../logs/{}/{}'.format(category, experiment_id)
    exists = os.path.exists(os.path.abspath(logdir))

    return logdir, exists


def parse_experiment_config(experiment_id, expdir='../../experiments/'):
    config = read_experiment_config(experiment_id, expdir=expdir)
    return parse_config_parameters(config)


def read_experiment_config(experiment_id, expdir='../../experiments/'):
    path = expdir + experiment_id + '.ini'
    config = configparser.ConfigParser()
    read_ok = config.read(path)

    if len(read_ok) == 0:
        raise Exception('Could not read experiment config from "{}", no such file or directory.'.format(path))

    config.set('environment', 'experiment_id', experiment_id)

    return config


def save_experiment_config(config, experiment_id, expdir='../../experiments/'):
    with open(expdir + experiment_id + '.ini', 'w') as fp:
        config.write(fp)


def str2bool(v):
    return v.lower() in ("True", "true")


def parse_config_parameters(config):
    # Mandatory parameters

    params = {
        'experiment_series': config['general'].get('experiment_series'),
        'question': config['general'].get('question'),
        'hypothesis': config['general'].get('hypothesis'),
        'remarks': config['general'].get('remarks'),

        'experiment_id': config['environment'].get('experiment_id'),
        'model_id': config['environment'].get('model_id'),
        'random_seed': config['environment'].getint('random_seed'),
        'preprocessor_id': config['environment'].get('preprocessor_id'),
        'splitter_id': config['environment'].get('splitter_id'),
        'evaluator_id': config['environment'].get('evaluator_id'),
        'explanator_id': config['environment'].get('explanator_id'),
        'gpu_id': config['environment'].get('gpu_id'),
        'loglevel': config['environment'].get('loglevel'),

        'dataset_id': config['data'].get('dataset_id'),
        'split_id': config['data'].get('split_id'),
        'leads_to_use': config['data'].get('leads_to_use').split(','),
        'clinical_parameters_outputs': config['data'].get('clinical_parameters_outputs').split(','),
        'subsampling_factor': config['data'].getint('subsampling_factor'),
        'subsampling_window_size': config['data'].getint('subsampling_window_size'),
        'clinical_parameters_inputs': config['data'].get('clinical_parameters_inputs'),
        'snapshot_id': config['data'].get('snapshot_id'),
        'record_ids_excluded': config['data'].get('record_ids_excluded'),
        'metadata_id': config['data'].get('metadata_id'),
        'stratification_variable': config['data'].get('stratification_variable'),
        'ratio_split': config['data'].getfloat('ratio_split'),
        'ratio_test': config['data'].getfloat('ratio_test'),

        'number_epochs': config['hyperparameters_general'].getint('number_epochs'),
        'optimizer': config['hyperparameters_general'].get('optimizer'),
        'learning_rate': config['hyperparameters_general'].getfloat('learning_rate'),
        'learning_rate_decay': config['hyperparameters_general'].getfloat('learning_rate_decay'),
        'shuffle': config['hyperparameters_general'].getboolean('shuffle'),
        'loss_function': config['hyperparameters_general'].get('loss_function'),
        'number_training_repetitions': config['hyperparameters_general'].getint('number_training_repetitions'),
        'validation_type': config['hyperparameters_general'].get('validation_type'),
        'batch_size': config['hyperparameters_general'].get('batch_size'),

        'metrics': config['evaluation'].get('metrics').split(','),
        'calculation_methods': config['evaluation'].get('calculation_methods').split(','),
        'class_names': config['evaluation'].get('class_names').split(','),
        'target_metric': config['evaluation'].get('target_metric'),
        'tensorboard_subdir': config['evaluation'].get('tensorboard_subdir'),
        'sensitivity_threshold': config['evaluation'].getfloat('sensitivity_threshold'),
        'specificity_threshold': config['evaluation'].getfloat('specificity_threshold'),
        'save_raw_results': config['evaluation'].getboolean('save_raw_results')
    }

    # Optional param group: flexible ecg model (these parameters are parsed differently, based on lists)
    try:
        params_ecgmodel_flex = {
            'ecgmodel_number_filters_conv': [int(x) for x in
                                             config['hyperparameters_ecgmodel_flex'].get('number_filters_conv').split(
                                                 ',')],
            'ecgmodel_number_neurons_dense': [int(x) for x in
                                              config['hyperparameters_ecgmodel_flex'].get('number_neurons_dense').split(
                                                  ',')],
            'ecgmodel_size_kernel_conv': [int(x) for x in
                                          config['hyperparameters_ecgmodel_flex'].get('size_kernel_conv').split(',')],
            'ecgmodel_size_kernel_pool': [int(x) for x in
                                          config['hyperparameters_ecgmodel_flex'].get('size_kernel_pool').split(',')],
            'ecgmodel_stride_conv': [int(x) for x in
                                     config['hyperparameters_ecgmodel_flex'].get('stride_conv').split(',')],
            'ecgmodel_stride_pool': [int(x) for x in
                                     config['hyperparameters_ecgmodel_flex'].get('stride_pool').split(',')],
            'ecgmodel_padding_conv': config['hyperparameters_ecgmodel_flex'].get('padding_conv').split(','),
            'ecgmodel_maxpooling_conv': [str2bool(x) for x in
                                         config['hyperparameters_ecgmodel_flex'].get('maxpooling_conv').split(',')],
            'ecgmodel_batchnorm_conv': [str2bool(x) for x in
                                        config['hyperparameters_ecgmodel_flex'].get('batchnorm_conv').split(',')],
            'ecgmodel_batchnorm_dense': [str2bool(x) for x in
                                         config['hyperparameters_ecgmodel_flex'].get('batchnorm_dense').split(',')],
            'ecgmodel_transition_conv_dense': config['hyperparameters_ecgmodel_flex'].get('transition_conv_dense'),
            'ecgmodel_dropout_rate_conv': [float(x) for x in
                                           config['hyperparameters_ecgmodel_flex'].get('dropout_rate_conv').split(',')],
            'ecgmodel_dropout_rate_dense': [float(x) for x in
                                            config['hyperparameters_ecgmodel_flex'].get('dropout_rate_dense').split(
                                                ',')],
            'ecgmodel_activation_function_conv': config['hyperparameters_ecgmodel_flex'].get(
                'activation_function_conv').split(','),
            'ecgmodel_activation_function_dense': config['hyperparameters_ecgmodel_flex'].get(
                'activation_function_dense').split(',')
        }
        params.update(params_ecgmodel_flex)
    except:
        pass

    # Optional parameter processing and consistency check
    if params['record_ids_excluded'] is not None:
        params['record_ids_excluded'] = params['record_ids_excluded'].split(',')

    if params['clinical_parameters_inputs'] is not None:
        params['clinical_parameters_inputs'] = params['clinical_parameters_inputs'].split(',')

    if params['validation_type'] == 'cross_validation':
        params['folds_cross_validation'] = config['hyperparameters_general'].getint('folds_cross_validation')
        if params['folds_cross_validation'] is None:
            raise Exception('Missing parameter "folds_cross_validation". Aborting.')

    if params['validation_type'] == 'bootstrapping':
        params['bootstrapping_n'] = config['hyperparameters_general'].getint('bootstrapping_n')
        if params['bootstrapping_n'] is None:
            raise Exception('Missing parameter "bootstrapping_n". Aborting.')

    return params


def extract_different_config_parameters(experiments):
    rows = []
    parameters = []

    for e in experiments:
        rows.append(parse_experiment_config(e))

    df = pd.DataFrame(rows)

    for c in df:
        variants = list(df[c])
        types = [type(x) for x in variants]

        if list in types:
            tmp = []
            for v in variants:
                tmp.append(str(v))
            variants = tmp

        if len(set(variants)) > 1:
            parameters.append(c)

    return parameters


def convert_suffix_to_dict(suffix):
    parts = suffix.split('_')
    kvpairs = {}

    for part in parts:
        key = part[0]
        value = part[1:]

        try:
            kvpairs[key] = int(value)
        except ValueError:
            pass

    return kvpairs
