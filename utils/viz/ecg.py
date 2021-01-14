import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

from sandbox.test_chronicle_db import select_r_r_intervals_from_ecg_on_cronicle_db_v1, \
    select_r_r_intervals_from_ecg_on_cronicle_db_v2
from utils.api.redcap_api import load_file
from utils.data.data import convert_lead_dict_to_matrix, load_ecg_xml, parse_ecg_xml, scale_ecg, derive_ecg_variants, \
    load_ecgs_from_ptbxl, load_ecg_csv, load_dataset, convert_matrix_to_lead_dict_ecg
from utils.file.file import cleanup_directory
from utils.misc.datastructure import perform_shape_switch
from utils.misc.logger import print_progress_bar
from utils.viz.multicolor import plot_multicolored_line


def plot_ecg(ecg, subdict='ecg_raw', title='ECG', save_to=None, colorize=None, normalize_colors=True,
             cmap='ColdDarkHot', cmap_min=0, cmap_max=1, colorbar_label=None, colorbar_tickvalues=None, columns=2,
             line_width=0.75):
    lead_names = [x for x in ecg[subdict]]
    sampling_rate = ecg['metadata']['sampling_rate_sec']
    assert ecg['metadata']['unitofmeasurement'] == 'mV'

    ecg_matrix = convert_lead_dict_to_matrix(ecg[subdict], shape_switch=False)

    if colorize is True:
        colorize = ecg_matrix

    if colorize is not None and normalize_colors is True:
        abs_max = np.percentile(np.abs(colorize), 100)
        abs_min = -abs_max
        norm = plt.Normalize(abs_min, abs_max)
    else:
        norm = None

    generate_medical_ecg_plot(ecg_matrix, sample_rate=sampling_rate, title=title, columns=columns,
                              lead_index=lead_names, style='bw', line_width=line_width, save_to=save_to,
                              color_matrix=colorize, norm=norm, cmap=cmap, cmap_min=cmap_min, cmap_max=cmap_max,
                              colorbar_label=colorbar_label, colorbar_tickvalues=colorbar_tickvalues)


def plot_ecg_from_matrix(ecg, metadata, lead_names, title='ECG', save_to=None, colorize=None, normalize_colors=True,
             cmap='ColdDarkHot', cmap_min=0, cmap_max=1, colorbar_label=None, colorbar_tickvalues=None, columns=2,
             line_width=0.75):

    sampling_rate = metadata['sampling_rate_sec']
    assert metadata['unitofmeasurement'] == 'mV'

    if colorize is True:
        colorize = ecg

    if colorize is not None and normalize_colors is True:
        abs_max = np.percentile(np.abs(colorize), 100)
        abs_min = -abs_max
        norm = plt.Normalize(abs_min, abs_max)
    else:
        norm = None

    generate_medical_ecg_plot(ecg, sample_rate=sampling_rate, title=title, columns=columns,
                              lead_index=lead_names, style='bw', line_width=line_width, save_to=save_to,
                              color_matrix=colorize, norm=norm, cmap=cmap, cmap_min=cmap_min, cmap_max=cmap_max,
                              colorbar_label=colorbar_label, colorbar_tickvalues=colorbar_tickvalues)


def plot_raw_ecg_and_delta_ecg(ecg, title='ECG', columns=2):
    lead_names = [x for x in ecg['ecg_raw']]
    all_lead_names = []
    sampling_rate = ecg['metadata']['sampling_rate_sec']
    lead_dict = {}

    for lead_name in lead_names:
        lead_dict[lead_name] = ecg['ecg_raw'][lead_name]
        lead_dict[lead_name + '_delta'] = ecg['ecg_delta'][lead_name]
        all_lead_names.append(lead_name)
        all_lead_names.append(lead_name + '_delta')

    ecg_matrix = convert_lead_dict_to_matrix(lead_dict, shape_switch=False)

    generate_medical_ecg_plot(ecg_matrix, sample_rate=sampling_rate, title=title, columns=columns,
                              lead_index=all_lead_names, style='bw', line_width=0.75)


def generate_medical_ecg_plot(
        ecg,
        sample_rate=500,
        title='ECG',
        lead_index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        lead_order=None,
        style=None,
        columns=2,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=False,
        line_width=0.5,
        color_matrix=None,
        save_to=None,
        norm=None,
        cmap='ColdDarkHot',
        cmap_min=0,
        cmap_max=1,
        colorbar_label=None,
        colorbar_tickvalues=None
):
    """Plot multi lead ECG chart.
    Code based on https://github.com/dy1901/ecg_plot
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    x_stride = 0.2
    y_stride = 0.5

    if not lead_order:
        lead_order = list(range(0, len(ecg)))
    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = ceil(leads / columns)
    display_factor = 1.25
    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor))
    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0,  # the left side of the subplots of the figure
        right=1,  # the right side of the subplots of the figure
        bottom=0,  # the bottom of the subplots of the figure
        top=1
    )

    fig.suptitle(title, y=0.995)

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if style == 'bw':
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line = (0, 0, 0)

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, x_stride))
        ax.set_yticks(np.arange(y_min, y_max, y_stride))

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for c in range(0, columns):
        for i in range(0, rows):
            if c * rows + i < leads:
                y_offset = -(row_height / 2) * ceil(i % rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0

                if c > 0:
                    x_offset = secs * c

                    if show_separate_line:
                        ax.plot([x_offset, x_offset],
                                [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3],
                                linewidth=line_width * display_factor, color=color_line)

                t_lead = lead_order[c * rows + i]

                step = 1.0 / sample_rate

                if show_lead_name:
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)

                # ax.plot(
                #     np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                #     ecg[t_lead] + y_offset,
                #     linewidth=line_width * display_factor,
                #     color=color_line
                # )

                if color_matrix is not None:
                    colors_lead = color_matrix[t_lead]
                else:
                    colors_lead = None

                plot_multicolored_line(
                    x=np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                    y=ecg[t_lead] + y_offset,
                    z=colors_lead,
                    ax=ax,
                    cmap=cmap,
                    linewidth=line_width * display_factor,
                    default_color=color_line,
                    norm=norm,
                    adjustplot=False
                )

    # Add bottom label
    ax.text(0.07, y_offset - 1.2, '25 mm/s, 10 mm/mV, recording length: {:.1f} s'.format(secs),
            fontsize=9 * display_factor)

    # Add colorbar at bottom
    if color_matrix is not None and colorbar_label is not None:
        if colorbar_tickvalues is not None:
            colorbar_ticklabels = {x: '{:.1f} s'.format(x / 500) for x in colorbar_tickvalues}
            ticks = list(colorbar_ticklabels.keys())
        else:
            ticks = None

        cax = fig.add_axes([0.175, 0.027, 0.1, 0.0075])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin=cmap_min, vmax=cmap_max), cmap=cmap),
                          orientation='horizontal', cax=cax, ticks=ticks)
        cb.ax.tick_params(labelsize=6 * display_factor)
        cb.set_label(colorbar_label, labelpad=-25, fontsize=7 * display_factor)

        if colorbar_tickvalues is not None:
            cb.ax.set_xticklabels([colorbar_ticklabels[k] for k in colorbar_ticklabels])

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
        plt.close()


# TODO: Standardize parameters for all plot_ecg_from_* functions
def plot_ecg_from_xml(path, title=None, scale=None, save_to=None, colorize=None, normalize_colors=True,
                      cmap='ColdDarkHot', cmap_min=0, cmap_max=1, colorbar_label=None, colorbar_tickvalues=None):
    leads, metadata = load_ecg_xml(path)
    ecg = {'leads': leads, 'metadata': metadata}
    scaled_ecg = scale_ecg(ecg, 1 / 1000)
    derived_ecg = derive_ecg_variants(scaled_ecg, ['ecg_raw'])

    if title is None:
        parts = path.split('/')
        title = parts[-1:][0]

    if save_to == 'same':
        save_to = path.replace('.xml', '.pdf')

    plot_ecg(derived_ecg, title=title, save_to=save_to, colorize=colorize, normalize_colors=normalize_colors, cmap=cmap,
             cmap_min=cmap_min, cmap_max=cmap_max, colorbar_label=colorbar_label,
             colorbar_tickvalues=colorbar_tickvalues)


def plot_ecg_from_redcap_api(record_id, title='ECG'):
    xmlcode = load_file(record_id=record_id, field='varid_ekg_hl7', event='baseline_arm_1')
    leads, metadata = parse_ecg_xml(xmlcode)
    ecg = {'leads': leads, 'metadata': metadata}
    scaled_ecg = scale_ecg(ecg, 1 / 1000)
    derived_ecg = derive_ecg_variants(scaled_ecg, ['ecg_raw'])
    plot_ecg(derived_ecg, title=title)


def plot_ecg_from_ptbxl(record_id, snapshot_id='1.0.1', title=None):
    if title is None:
        title = record_id

    ecgs = load_ecgs_from_ptbxl(snapshot_id, snapshot_directory='../data/ptbxl/snapshots')
    derived_ecg = derive_ecg_variants(ecgs[record_id], ['ecg_raw'])
    plot_ecg(derived_ecg, title=title)


def plot_ecg_from_csv(path, title=None, scale=None, save_to=None, colorize=None, normalize_colors=True,
                      cmap='ColdDarkHot', cmap_min=0, cmap_max=1, colorbar_label=None, colorbar_tickvalues=None, columns=2):
    leads, metadata = load_ecg_csv(path)
    ecg = {'leads': leads, 'metadata': metadata}

    if title is None:
        parts = path.split('/')
        title = parts[-1:][0]

    if scale is not None:
        ecg = scale_ecg(ecg, scale)

    if save_to == 'same':
        save_to = path.replace('.csv', '.pdf')

    derived_ecg = derive_ecg_variants(ecg, ['ecg_raw'])
    plot_ecg(derived_ecg, title=title, save_to=save_to, colorize=colorize, columns=columns, normalize_colors=normalize_colors,
             cmap=cmap, cmap_min=cmap_min, cmap_max=cmap_max, colorbar_label=colorbar_label,
             colorbar_tickvalues=colorbar_tickvalues)


def plot_ecg_from_dataset(record_id, dataset_id, title=None, colorize=None):
    if title is None:
        title = record_id

    records = load_dataset(dataset_id, dataset_directory='../data/datasets/')

    plot_ecg(records[record_id], title=title, colorize=colorize)


def calc_color_matrix_for_r_r_intervals_v1(cronicle_db_record_id, lead_ids, lead_length=5000):
    collected = []

    for lead_id in lead_ids:
        # Perform RR interval segmentation only for lead I, then propagate values to all leads
        if lead_id == 'lead_I':
            c_lead = np.zeros((lead_length,))
            r_r_intervals = select_r_r_intervals_from_ecg_on_cronicle_db_v1(lead_id, cronicle_db_record_id,
                                                                            threshold=300)

            if len(r_r_intervals) > 0:
                print(lead_id, len(r_r_intervals))
                r_r_distances = [x['R_R_DISTANCE'] for x in r_r_intervals]
                r_r_max = max(r_r_distances)

                for r_r_interval in r_r_intervals:
                    r_r_distance_normalized = r_r_interval['R_R_DISTANCE'] / r_r_max
                    r_r_interval_start = r_r_interval['R_R_START']
                    r_r_interval_end = r_r_interval['R_R_END']

                    for i in np.arange(start=r_r_interval_start, stop=r_r_interval_end):
                        c_lead[i] = r_r_distance_normalized

                    # # Mark R peaks for testing
                    # for i in np.arange(start=r_r_interval_start, stop=r_r_interval_start + 50):
                    #     c_lead[i] = 1
                    # for i in np.arange(start=r_r_interval_end, stop=r_r_interval_end + 50):
                    #     c_lead[i] = 1

            # c_lead[0] = -1
            # c_lead[lead_length-1] = 1

        collected.append(c_lead)

    return np.array(collected), r_r_max


def calc_color_matrix_for_r_r_intervals_v2(cronicle_db_record_id, lead_ids, lead_length=5000):
    collected = []

    for lead_id in lead_ids:
        # Perform RR interval segmentation only for lead I, then propagate values to all leads
        if lead_id == 'lead_I':
            c_lead = np.zeros((lead_length,))
            r_r_intervals = select_r_r_intervals_from_ecg_on_cronicle_db_v2(lead_id, cronicle_db_record_id,
                                                                            threshold=300)

            if len(r_r_intervals) > 0:
                print(lead_id, len(r_r_intervals))
                r_r_distances = [x['R_R_DISTANCE'] for x in r_r_intervals]
                r_r_mean = np.mean(r_r_distances)
                r_r_deviations = [x['R_R_DISTANCE'] - r_r_mean for x in r_r_intervals]
                r_r_deviation_max = max(np.abs(r_r_deviations))
                r_r_deviation_min = r_r_deviation_max * -1
                r_r_stddev = np.std(r_r_distances)
                print('r_r_mean', r_r_mean)

                for r_r_interval in r_r_intervals:
                    r_r_deviation = r_r_interval['R_R_DISTANCE'] - r_r_mean
                    r_r_deviation_normalized = r_r_deviation / r_r_deviation_max
                    r_r_interval_start = r_r_interval['R_R_START']
                    r_r_interval_end = r_r_interval['R_R_END']

                    for i in np.arange(start=r_r_interval_start, stop=r_r_interval_end):
                        if np.abs(r_r_deviation) > r_r_stddev:
                            c_lead[i] = r_r_deviation_normalized

        collected.append(c_lead)

    return np.array(collected), r_r_deviation_max, r_r_deviation_min


def calc_color_matrix_for_r_r_intervals_v3(cronicle_db_record_id, lead_ids, lead_length=5000):
    collected = []

    for lead_id in lead_ids:
        # Perform RR interval segmentation only for lead I, then propagate values to all leads
        if lead_id == 'lead_I':
            c_lead = np.zeros((lead_length,))
            r_r_intervals = select_r_r_intervals_from_ecg_on_cronicle_db_v2(lead_id, cronicle_db_record_id,
                                                                            threshold=300)

            if len(r_r_intervals) > 0:
                print(lead_id, len(r_r_intervals))
                r_r_distances = [x['R_R_DISTANCE'] for x in r_r_intervals]
                r_r_mean = np.mean(r_r_distances)
                r_r_deviations = [x['R_R_DISTANCE'] - r_r_mean for x in r_r_intervals]
                r_r_deviation_max = max(np.abs(r_r_deviations))
                r_r_deviation_min = r_r_deviation_max * -1
                r_r_stddev = np.std(r_r_distances)
                print('r_r_mean', r_r_mean)

                for r_r_interval in r_r_intervals:
                    r_r_deviation = r_r_interval['R_R_DISTANCE'] - r_r_mean
                    r_r_interval_start = r_r_interval['R_R_START']
                    r_r_interval_end = r_r_interval['R_R_END']

                    for i in np.arange(start=r_r_interval_start, stop=r_r_interval_end):
                        if np.abs(r_r_deviation) > r_r_stddev:
                            c_lead[i] = 1

        collected.append(c_lead)

    return np.array(collected)


def generate_color_matrix_for_single_interval(n_leads, len_leads, color_from, color_to):
    mtrx = np.zeros((n_leads, len_leads))

    for t in np.arange(start=color_from, stop=color_to):
        for n in range(n_leads):
            mtrx[n][t] = 1

    return mtrx


def plot_extracted_features_clusters(ex_ecgparts, ex_heatmaps, cluster_labels, nrows=30, ncols=30):
    H = {x: [] for x in set(cluster_labels)}
    E = {x: [] for x in set(cluster_labels)}

    for c, e, h in zip(cluster_labels, ex_ecgparts, ex_heatmaps):
        H[c].append(h)
        E[c].append(e)

    for c in set(cluster_labels):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        axes = np.array(ax).flatten()

        for cax, e, h in zip(axes, E[c], H[c]):
            cax.plot(e)

        for a in axes:
            a.set_yticks([])
            a.set_xticks([])

        plt.tight_layout()
        print(c)
        plt.savefig('/home/nils/Desktop/cluster_{}.png'.format(c))
        plt.close()
        plt.clf()


def plot_extracted_features(ex_ecgparts, ex_heatmaps, onlypaths=False, figsize=(2, 10), linewidth=5, cellar='/tmp'):
    paths = []

    if not onlypaths:
        cleanup_directory('{}/featureplots'.format(cellar), make=True)

    fig, ax = plt.subplots(figsize=figsize)

    for e, h, i in zip(ex_ecgparts, ex_heatmaps, range(len(ex_ecgparts))):
        path = '{}/featureplots/feature_{}.png'.format(cellar, i)
        paths.append(path)

        if onlypaths==False:
            abs_max = np.percentile(np.abs(h), 100)
            norm = plt.Normalize(-abs_max, abs_max)

            lc = plot_multicolored_line(y=e, z=h, ax=ax, linewidth=linewidth, norm=norm, adjustax=True)

            if max(e) - min(e) < 0.7:
                m = np.mean(e)
                ax.set_ylim(m - 0.5, m + 0.5)
            else:
                ax.set_ylim(min(e), max(e))

            ax.set_yticks([])
            ax.set_xticks([])

            plt.tight_layout()
            plt.savefig(path)
            lc.remove()

            print_progress_bar(i, len(ex_ecgparts))

    plt.clf()
    plt.close()

    return paths


def plot_all_csvs_in_dir_as_pdf(directory):
    files = os.listdir(directory)

    for file in files:
        if file.endswith('.csv'):
            try:
                print('Plotting "{}"'.format(file))
                plot_ecg_from_csv(path='{}/{}'.format(directory, file),
                                  scale=1 / 1000,
                                  save_to='same',
                                  columns=2
                                  )
            except Exception as e:
                print(e)
