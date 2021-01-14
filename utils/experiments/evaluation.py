import logging
import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.saving.model_config import model_from_json

from dataloading.basic_dataloader import BasicDataloader
from utils.data.data import derive_binary_one_hot_classes_for_list_of_labels, load_dataset, subsample_ecgs, load_split
from utils.experiments.metrics import auc, sroc, confusionmatrix, sensitivity, specificity, roc, \
    calculate_metrics_for_predictions, get_max_value_for_metric
from utils.experiments.model import load_model, load_model_and_weights_from_paths
from utils.experiments.validation import single_repeated, cross_validation_repeated, bootstrapping_repeated, \
    bootstrapping, cross_validation, derive_validation_method_from_experiment_config
from utils.file.file import parse_experiment_config, load_dict_from_json, combine_pdfs, save_string_to_file, \
    load_string_from_file, extract_different_config_parameters, unpickle_data, pickle_data
from utils.misc.datastructure import perform_shape_switch


def load_results_for_epoch(exp_id, epoch_id, class_name, calculation_method):
    path = '../../logs/experiments/{}/results_{}.json'.format(exp_id, epoch_id)
    results = load_dict_from_json(path)

    return results[class_name][calculation_method]


def get_experiment_ids(experiment_id):
    validation_method = derive_validation_method_from_experiment_config(experiment_id)
    params = parse_experiment_config(experiment_id)

    experiment_ids = []

    if validation_method == 'repeated_single':
        experiment_ids = single_repeated(experiment_id, params['number_training_repetitions'], None, run=False)
    elif validation_method == 'single':
        experiment_ids = [experiment_id]

    elif validation_method == 'repeated_cross_validation':
        experiment_ids = cross_validation_repeated(experiment_id, params['folds_cross_validation'],
                                                   params['number_training_repetitions'], params['split_id'], None,
                                                   run=False)
    elif validation_method == 'cross_validation':
        experiment_ids = cross_validation(experiment_id, params['folds_cross_validation'], params['split_id'], None,
                                          run=False)

    elif validation_method == 'repeated_bootstrapping':
        experiment_ids = bootstrapping_repeated(experiment_id, params['bootstrapping_n'],
                                                params['number_training_repetitions'], params['split_id'], None,
                                                run=False)
    elif validation_method == 'bootstrapping':
        experiment_ids = bootstrapping(experiment_id, params['bootstrapping_n'], params['split_id'], None, run=False)

    else:
        raise Exception('Unknown validation method "{}". Aborting.'.format(validation_method))

    assert len(experiment_ids) > 0

    return experiment_ids


def derive_static_values_from_experiment_id(experiment_id, mainsep='__', subsep='_'):
    static_values = {}

    try:
        statics = experiment_id.rsplit(mainsep, maxsplit=1)[1]
        parts = statics.split(subsep)

        for part in parts:
            key = part[0]
            value = int(part[1:])
            static_values[key] = value

    except IndexError:  # If experiment_id does not contain static values at the end, return empty dict
        pass

    return static_values


def create_line(results_raw, static_values):
    line = {}

    results_metrics = results_raw['metrics']
    results_classification = results_raw['classification']

    for s in static_values:
        line[s] = static_values[s]

    for m in results_metrics:
        line[m] = results_metrics[m]

    for c in results_classification:
        line['{}_classification'.format(c)] = results_classification[c]

    line['static_values'] = static_values

    return line


def cleanup_weights_for_experiment(experiment_id, class_names, calculation_methods, epochs):
    logging.info('Cleaning up weights')
    logdir = '../../logs/experiments'

    not_to_delete = []

    for class_name in class_names:
        for calculation_method in calculation_methods:
            subdir = '{}/{}/{}/{}'.format(logdir, experiment_id, class_name, calculation_method)
            df = pd.read_excel('{}/results_filtered.xlsx'.format(subdir))
            not_to_delete += list(df['static_values'])

    not_to_delete = set(not_to_delete)

    experiment_ids = get_experiment_ids(experiment_id)

    for exp_id in experiment_ids:
        static_values = derive_static_values_from_experiment_id(exp_id)

        for e in range(epochs):
            static_values['e'] = e
            path = '{}/{}/weights_e{}.h5'.format(logdir, exp_id, e)

            if str(static_values) not in not_to_delete:
                try:
                    os.remove(path)
                    logging.debug('Removed {}'.format(path))
                except FileNotFoundError:
                    pass
            else:
                logging.info('Keeping {}'.format(path))


def derive_weights_and_model_paths_for_experiment(experiment_id, class_name, calculation_method, epochs, logdir='../../logs/experiments'):
    subdir = '{}/{}/{}/{}'.format(logdir, experiment_id, class_name, calculation_method)
    df = pd.read_excel('{}/results_filtered.xlsx'.format(subdir))
    sv = list(df['static_values'])

    experiment_ids = get_experiment_ids(experiment_id)

    weights_paths = []
    model_paths = []

    for exp_id in experiment_ids:
        static_values = derive_static_values_from_experiment_id(exp_id)

        for e in range(epochs):
            static_values['e'] = e
            weights_path = '{}/{}/weights_e{}.h5'.format(logdir, exp_id, e)
            model_path = '{}/{}/model.json'.format(logdir, exp_id)

            if str(static_values) in sv:
                weights_paths.append(weights_path)
                model_paths.append(model_path)

    return weights_paths, model_paths


def perform_test_for_models(experiment_id, params, logdir='../../logs/experiments', datasetdir='../../data/datasets/'):
    try:
        load_split(params['split_id'] + '_test')
    except FileNotFoundError:
        logging.info('No test split found')
        return None

    logging.info('Testing models')

    record_ids, metadata, diagnosis, clinical_parameters, ecg_raw = BasicDataloader(params).load_test_data()

    # X = [np.asarray(ecg_raw), np.asarray(clinical_parameters)] # TODO: make dynamic for combined model
    X = np.asarray(ecg_raw)
    Y = np.asarray(diagnosis)

    print('X shape: ', np.shape(X))

    for class_name in params['class_names']:
        for calculation_method in params['calculation_methods']:
            subdir = '{}/{}/{}/{}'.format(logdir, experiment_id, class_name, calculation_method)

            weights_paths, model_paths = derive_weights_and_model_paths_for_experiment(experiment_id,
                                                                                       class_name,
                                                                                       calculation_method,
                                                                                       epochs=params['number_epochs'],
                                                                                       logdir=logdir)

            results = []

            for mp, wp in zip(model_paths, weights_paths):
                clear_session()
                res = evaulate_model(
                    modelpath=mp,
                    weightspath=wp,
                    X=X,
                    Y=Y,
                    record_ids=record_ids,
                    class_name=class_name,
                    calculation_method=calculation_method,
                    metrics=params['metrics'])

                results.append(res)

            df = pd.DataFrame(results)
            df.to_excel('{}/test_results_single.xlsx'.format(subdir))
            logging.info('Saved single results as .xlsx')

            logging.info('Calculating ensemble scores')

            for ensembling_method in params['ensembling_methods']:
                res = evaulate_models_as_ensemble(
                    modelpaths=model_paths,
                    weightspaths=weights_paths,
                    X=X,
                    Y=Y,
                    record_ids=record_ids,
                    class_name=class_name,
                    calculation_method=calculation_method,
                    metrics=params['metrics'],
                    ensembling_method=ensembling_method)

                df = pd.DataFrame([res])
                df.to_excel('{}/test_results_ensemble_{}.xlsx'.format(subdir, ensembling_method))
                logging.info('Saved ensemble results ({}) as .xlsx'.format(ensembling_method))


def load_results_for_experiment(experiment_id, class_name, calculation_method):
    experiment_ids = get_experiment_ids(experiment_id)
    params = parse_experiment_config(experiment_id)

    lines = []

    for exp_id in experiment_ids:
        static_values = derive_static_values_from_experiment_id(exp_id)

        for e in range(params['number_epochs']):
            epoch_id = 'e{}'.format(e)
            results_raw = load_results_for_epoch(exp_id, epoch_id, class_name, calculation_method)
            static_values['e'] = e
            lines.append(create_line(results_raw, dict(static_values)))

    return pd.DataFrame(lines)


def filter_df_based_on_metric_thresholds(df, metric_thresholds):
    if metric_thresholds is not None:
        for m in metric_thresholds:
            df = df[df[m] > metric_thresholds[m]]

    return df


def filter_df_based_on_max_value(df, columnname, maxvalue):
    df = df[df[columnname] <= maxvalue]

    return df


def filter_df_based_on_group_column(df, target_metric, group_column, groups):
    lines = []

    for g in groups:
        sub_df = df[df[group_column] == g]

        if len(sub_df) < 1:
            raise Exception(
                'Your metric thresholds are too restrictive. It was not possible to select one row for group {} in column {}'.format(
                    g, group_column))

        sub_df = sub_df.sort_values(by=target_metric, ascending=False)
        sub_df = sub_df.head(1)
        lines.append(sub_df)

    df = pd.concat(lines)

    if len(df) != len(groups):
        raise Exception(
            'Your metric thresholds are too restrictive. It was not possible to select one row for each {} ({})'.format(
                group_column, groups))

    return df


def filter_results_based_on_metrics(df, experiment_id, target_metric, metric_thresholds):
    validation_method = derive_validation_method_from_experiment_config(experiment_id)

    df_f = filter_df_based_on_metric_thresholds(df, metric_thresholds)

    if validation_method == 'single' or validation_method == None:
        pass
    elif validation_method == 'repeated_single':
        df_f = df_f.sort_values(by=target_metric, ascending=False)
        df_f = df_f.head(1)

    elif validation_method in ['cross_validation', 'repeated_cross_validation']:
        df_f = filter_df_based_on_group_column(df_f, target_metric, 'k', set(df['k']))

    elif validation_method in ['repeated_bootstrapping', 'bootstrapping']:
        df_f = filter_df_based_on_group_column(df_f, target_metric, 'n', set(df['n']))

    else:
        raise Exception('Unknown validation method "{}". Aborting.'.format(validation_method))

    return df_f


def create_classification_table(df):  # TODO: prüfen, ob mit allen validierungsverfahren kompatibel
    TP = [val for sublist in df['TP_classification'] for val in sublist]
    TN = [val for sublist in df['TN_classification'] for val in sublist]
    FP = [val for sublist in df['FP_classification'] for val in sublist]
    FN = [val for sublist in df['FN_classification'] for val in sublist]
    all_ids = set(TP + TN + FP + FN)

    lines = []

    for i in all_ids:
        lines.append({'record': i, 'TP': TP.count(i), 'FP': FP.count(i), 'TN': TN.count(i), 'FN': FN.count(i)})

    return pd.DataFrame(lines)


def extract_rocs_from_df(df):
    rocs = list(df['ROC'])
    roc_titles = ['{}, AUC: {:.2f}'.format(str(s).replace('{', '').replace('}', '').replace("'", ''), a) for s, a in
                  zip(df['static_values'], df['AUC'])]

    return rocs, roc_titles


def plot_roc_to_pdf(roc, roc_title, path, lw=2):
    fprs = list(roc['FPR'])
    tprs = list(roc['TPR'])

    fprs.append(0.0)
    fprs.append(1.0)
    tprs.append(0.0)
    tprs.append(1.0)

    fprs = sorted(fprs)
    tprs = sorted(tprs)

    plt.figure()

    plt.plot(fprs, tprs, color='k', lw=lw)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_title)

    plt.savefig(path)
    plt.close()


def plot_rocs_to_pdf(rocs, roc_titles, path, filename):
    filepaths = []

    for roc, roc_title, i in zip(rocs, roc_titles, range(len(rocs))):
        filepath = '{}/roc{}.pdf'.format(path, i)
        plot_roc_to_pdf(roc, roc_title, filepath)
        filepaths.append(filepath)

    targetpath = '{}/{}'.format(path, filename)
    combine_pdfs(filepaths, targetpath, cleanup=True)


def plot_histogram_and_boxplot_for_metric_to_pdf(metric, values, filepath, n=None):
    fig, axs = plt.subplots(nrows=2, figsize=(20, 10))

    if n is None:
        n = len(values)
    mx = get_max_value_for_metric(metric)
    ticks = [round(0, 2), round(mx * 0.25, 2), round(mx * 0.5, 2), round(mx * 0.75, 2), round(mx, 2)]

    axs[0].hist(values, n, facecolor='g')
    axs[0].set_title('Distribution for {}'.format(metric))
    axs[0].set_xticks(ticks)
    axs[0].set_xlim((0, 1))

    axs[1].boxplot(values, showfliers=True, vert=False)
    axs[1].set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(filepath)


def plot_histograms_and_boxplots_to_pdf(df, metrics, path, filename, n=None):
    filepaths = []

    for metric in metrics:
        if metric != 'DOR':
            filepath = '{}/hist_boxpl_{}.pdf'.format(path, metric)
            plot_histogram_and_boxplot_for_metric_to_pdf(metric, list(df[metric].values), filepath, n=n)
            filepaths.append(filepath)

    targetpath = '{}/{}'.format(path, filename)
    combine_pdfs(filepaths, targetpath, cleanup=True)


def calculate_mean_values_for_filtered_results(df, metrics):
    means = {'value': 'mean'}
    stddevs = {'value': 'stddev'}

    for m in metrics:
        means[m] = np.mean(df[m])
        stddevs[m] = np.std(df[m])

    df_ret = pd.DataFrame([means, stddevs])
    df_ret = df_ret.set_index('value')

    return df_ret


def save_results_for_experiment(experiment_id, class_names, calculation_methods, metrics, target_metric,
                                metric_thresholds, save_raw_results):
    logdir = '../../logs/experiments/{}'.format(experiment_id)

    for class_name in class_names:
        for calculation_method in calculation_methods:

            # Create sub directories for classes and calculation methods
            subdir = '{}/{}/{}'.format(logdir, class_name, calculation_method)
            os.makedirs(subdir, exist_ok=True)

            # Load raw results from .json files into dataframe
            logging.info('Loading results from disk... ({})'.format(calculation_method))
            df = load_results_for_experiment(experiment_id, class_name, calculation_method)
            logging.info('Loading complete')

            if save_raw_results is True:
                logging.info('Saving as .xlsx')
                df.to_excel('{}/results_raw.xlsx'.format(subdir))
                logging.info('Saved raw results as .xlsx')
                plot_histograms_and_boxplots_to_pdf(df, metrics, subdir, 'metrics_histograms_boxplots_raw.pdf', n=100)
                logging.info('Saved metric histograms and boxplots as .pdf')

            # Filter results based on thresholds, target metric and validation method -> best n models
            df = filter_results_based_on_metrics(df, experiment_id, target_metric, metric_thresholds)
            logging.info('Filtered raw results based on metrics and thresholds')
            df.to_excel('{}/results_filtered.xlsx'.format(subdir))
            logging.info('Saved filtered results as .xlsx')

            # Calculate mean values for each metric
            df_mean = calculate_mean_values_for_filtered_results(df, metrics)
            logging.info('Calculated mean metrics')
            df_mean.to_excel('{}/mean_metrics.xlsx'.format(subdir))
            logging.info('Saved mean metrics as .xlsx')

            # Plot histograms and boxplots for each metric
            plot_histograms_and_boxplots_to_pdf(df, metrics, subdir, 'metrics_histograms_boxplots.pdf')
            logging.info('Saved metric histograms and boxplots as .pdf')

            # Aggregate classifications for each record based on filtered results (best n models)
            classification_table = create_classification_table(df)
            logging.info('Calculated classification table')
            classification_table.to_excel('{}/classification_table.xlsx'.format(subdir))
            logging.info('Saved classification table as .xlsx')

            # Calculate and plot SROC together with other ROC curves for each of the best n models
            s = sroc(df['TP'], df['FP'], df['TN'], df['FN'])
            sroc_auc = auc(s)
            rocs, roc_titles = extract_rocs_from_df(df)
            rocs.append(s)
            roc_titles.append('SROC, AUC: {:.2f}'.format(sroc_auc))
            logging.info('Calculated ROCs')
            plot_rocs_to_pdf(rocs, roc_titles, subdir, 'ROCs.pdf')
            logging.info('Saved ROCs as .pdf')


def distribute_results_for_experiment(experiment_id, class_names, calculation_methods, recipients):
    if recipients is not None:
        for class_name in class_names:
            for calculation_method in calculation_methods:
                send_experiment_results_as_email(experiment_id=experiment_id,
                                                 class_name=class_name,
                                                 calculation_method=calculation_method,
                                                 recipients=recipients)


def send_experiment_results_as_email(experiment_id,
                                     class_name,
                                     calculation_method,
                                     recipients,
                                     experiment_logdir='../../logs/experiments'):
    smtp_server = 'smtp.strato.de'
    port = 465
    sender = 'experiments@nilsgumpfer.com'
    password = '4b868e70d26138ef6469e446d1'
    context = ssl.create_default_context()
    exclude = {'results_raw.xlsx'}

    # TODO: consider email size restriction!

    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender, password)

        directory = '{}/{}/{}/{}'.format(experiment_logdir, experiment_id, class_name, calculation_method)

        msg = EmailMessage()
        msg['Subject'] = 'Results of experiment "{}" ({}, {})'.format(experiment_id, class_name, calculation_method)
        msg['To'] = recipients
        msg['From'] = sender

        filenames = list(set(os.listdir(directory)) - exclude)
        msgcontent = str(filenames).replace(',', '\n').replace('[', '').replace(']', '').replace("'", '').replace(' ',
                                                                                                                  '')
        msg.set_content(msgcontent)

        for filename in filenames:
            path = os.path.join(directory, filename)

            # Skip directories
            if not os.path.isfile(path):
                continue

            ctype, encoding = mimetypes.guess_type(path)

            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'

            maintype, subtype = ctype.split('/', 1)

            with open(path, 'rb') as fp:
                msg.add_attachment(fp.read(),
                                   maintype=maintype,
                                   subtype=subtype,
                                   filename=filename)

        server.send_message(msg)
        logging.info('Results of experiment "{}" ({}, {}) sent via email to {}'.format(experiment_id, class_name,
                                                                                       calculation_method, recipients))


def save_placeholder_files_to_logdir(experiment_id, placeholders):
    for p in placeholders:
        path = '../../logs/experiments/{}/{}.txt'.format(experiment_id, p)
        save_string_to_file('<empty>', path)


def get_logdir_filepaths_for_experiment(experiment_id, class_names, calculation_methods,
                                        base_path='../../logs/experiments'):
    filepaths = []

    modelvizpath = '{}/{}/model.png'.format(base_path, experiment_id)

    if os.path.exists(modelvizpath):
        filepaths.append(modelvizpath)

    for class_name in class_names:
        for calculation_method in calculation_methods:
            path = '{}/{}/{}/{}'.format(base_path, experiment_id, class_name, calculation_method)

            if os.path.exists(path):
                filenames = os.listdir(path)
                for f in filenames:
                    filepaths.append(
                        '{}/{}/{}/{}/{}'.format(base_path, experiment_id, class_name, calculation_method, f))

    return filepaths


def get_experiment_overview(columns, expdir='../experiments/', logdir='../logs/experiments'):
    # List all experiments from logdir, initialize dict
    filenames = natsorted(os.listdir(expdir))
    experiments = {}

    idcount = 0

    # Loop over all experiments
    for f in filenames:
        # Load experiment config
        exp_config = parse_experiment_config(f.replace('.ini', ''), expdir=expdir)

        # Derive experiment id and initialize values dict
        exp_id = exp_config['experiment_id']
        exp_values = {c: '' for c in columns}
        exp_values['experiment_id'] = exp_id
        exp_values['id'] = idcount

        # Derive relevant information from exp config
        for c in columns:
            try:
                if exp_config[c] is not None:
                    exp_values[c] = exp_config[c]
            except KeyError:
                pass

        # Derive date from logfile
        try:
            with open('{}/{}/logfile.log'.format(logdir, exp_id)) as logfile:
                first_line = logfile.readline()
                fl_s = first_line.split(' ', maxsplit=2)
                date = fl_s[0]
                time = fl_s[1].split(',')[0]
                exp_values['execution'] = '{} {}'.format(date, time)
        except FileNotFoundError:
            pass

        # Load conclusion and interpretation, assign to values dict
        for v in ['conclusion', 'interpretation']:
            try:
                exp_values[v] = load_string_from_file('{}/{}/{}.txt'.format(logdir, exp_id, v))
            except FileNotFoundError:
                pass

        exp_values['logdir_files'] = get_logdir_filepaths_for_experiment(exp_id, exp_config['class_names'],
                                                                         exp_config['calculation_methods'], logdir)

        # Assign values to experiment if it is a main experiment
        if f.find('__') == -1:
            experiments[exp_id] = exp_values

        # Derive main experiment, assign sub to main, and assign values to sub if it is a sub experiment
        else:
            main_exp = derive_main_experiment(exp_id)

            if 'subexperiments' not in experiments[main_exp].keys():
                experiments[main_exp]['subexperiments'] = []

            experiments[main_exp]['subexperiments'].append(exp_values)

        idcount += 1

    list_of_experiments = [experiments[e] for e in experiments]

    return list_of_experiments


def read_mean_metrics_file(experiment_id, calculation_method, classname):
    path = '../../logs/experiments/{}/{}/{}/mean_metrics.xlsx'.format(experiment_id, classname, calculation_method)
    df = pd.read_excel(path, index_col=0)
    return df.to_dict('records')[0]


def derive_main_experiment(exp_id):
    main_exp = exp_id.split('__', maxsplit=1)[0]
    return main_exp


def generate_overview_excel_sheet_for_experiments(experiments, metrics, calculation_method, classname, targetpath):
    rows = []
    parameters = extract_different_config_parameters(experiments)
    for e in experiments:
        row = {}

        config = parse_experiment_config(e)

        row['experiment_id'] = e

        for p in parameters:
            row[p] = config[p]

        values = read_mean_metrics_file(e, calculation_method, classname)

        for m in metrics:
            row[m] = round(values[m], 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('experiment_id')
    df.to_excel(targetpath)


def calculate_classifications(modelpath, epoch, inputs, outputs, record_ids, clinical_parameters_outputs, class_name):
    logging.info('Calculating classifications')

    model = load_model(modelpath, epoch)

    predictions = model.predict_on_batch(np.asarray(inputs))

    class_names = derive_binary_one_hot_classes_for_list_of_labels(clinical_parameters_outputs)
    c = list(class_names).index(class_name)

    tp, fp, tn, fn, cls = confusionmatrix(outputs, predictions, record_ids, c=c, calculation_method='subsample_level')

    classifications = []

    for r in record_ids:
        for k in ['TP', 'TN', 'FP', 'FN']:
            if r in cls[k]:
                classifications.append(k)

    return classifications


def evaulate_model(modelpath, weightspath, X, Y, record_ids, class_name, calculation_method, metrics, return_predictions=False):
    model = load_model_and_weights_from_paths(modelpath, weightspath)

    logging.info('Predicting on model ({}).'.format(weightspath))
    predictions = model.predict_on_batch(np.asarray(X))

    result = calculate_metrics_for_predictions(y_true=Y,
                                               y_pred=predictions,
                                               y_classes=[class_name],
                                               y_record_ids=record_ids,
                                               metrics_to_calculate=metrics,
                                               calculation_methods=[calculation_method],
                                               class_names_to_log=[class_name])

    if return_predictions:
        return result[class_name][calculation_method]['metrics'], predictions
    else:
        return result[class_name][calculation_method]['metrics']


def evaulate_models_as_ensemble(modelpaths, weightspaths, X, Y, record_ids, class_name, calculation_method, metrics, ensembling_method):
    predictions_model = []

    for i, modelpath, weightspath in zip(range(len(modelpaths)), modelpaths, weightspaths):
        clear_session()
        model = load_model_and_weights_from_paths(modelpath, weightspath)

        logging.info('Predicting on model {}/{} ({}).'.format(i, len(modelpaths) - 1, weightspath))

        predictions_model.append(model.predict_on_batch(np.asarray(X)))

    if ensembling_method == 'voting':
        predictions = voting(np.asarray(predictions_model))
    elif ensembling_method == 'averaging':
        predictions = np.mean(predictions_model, axis=0)
    else:
        raise Exception('Unknown ensembling method "{}"'.format(ensembling_method))

    result = calculate_metrics_for_predictions(y_true=Y,
                                               y_pred=predictions,
                                               y_classes=[class_name],
                                               y_record_ids=record_ids,
                                               metrics_to_calculate=metrics,
                                               calculation_methods=[calculation_method],
                                               class_names_to_log=[class_name])

    return result[class_name][calculation_method]['metrics']


def voting(model_predictions):
    sample_votes = []

    n_samples = len(model_predictions[0])
    n_classes = len(model_predictions[0][0])

    for s in range(n_samples):
        votes = np.zeros(n_classes)
        predictions = model_predictions[:, s, :]

        for pred in predictions:
            votes[np.argmax(pred)] += 1

        sample_vote = np.zeros(n_classes)
        sample_vote[np.argmax(votes)] = 1.0
        sample_votes.append(sample_vote)

    return np.asarray(sample_votes)


# model_predictions = []
#
# for i in range(100):
#     line = []
#     for s in range(2344):
#         pred = np.random.randint(low=0, high=100) / 100
#         line.append([pred, 1-pred])
#     model_predictions.append(line)
#
# vs = voting(np.asarray(model_predictions))
#
# for v in vs:
#     print(v)