import logging

# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from utils.data.data import load_dataset, save_crops
from utils.misc.logger import print_progress_bar


def find_free_space_between_peaks(peaks, min_width=2000):
    peaks = sorted(peaks)

    max_width = 0
    start = 0
    end = 0

    for i in range(len(peaks) - 1):
        s = peaks[i]
        e = peaks[i+1]

        dist = e - s

        if dist >= min_width and dist >= max_width:
            end = e
            start = s
            max_width = dist

    return start, end


def find_artefacts_and_crop_ecg(ecg, recid, thresh=5):
    # fig, axs = plt.subplots(nrows=13, figsize=(20, 10))
    # plt.suptitle(recid)

    lead_length = len(ecg[list(ecg.keys())[0]])
    safety_dist = int(lead_length / 100)

    all_peaks = [0, lead_length]

    # for lead_id, ax in zip(list(ecg.keys()), axs):
    for lead_id in list(ecg.keys()):
        lead = ecg[lead_id]
        inv_lead = lead * -1

        peaks = find_peaks(lead, height=np.mean(lead) + thresh)[0]
        all_peaks += list(peaks)

        neg_peaks = find_peaks(inv_lead, height=np.mean(inv_lead) + thresh)[0]
        all_peaks += list(neg_peaks)

        # y_peaks = [lead[x] for x in peaks]
        # y_neg_peaks = [lead[x] for x in neg_peaks]

        # ax.plot(range(len(lead)), lead, 'k', peaks, y_peaks, 'rx', neg_peaks, y_neg_peaks, 'rx')
        # ax.set_yticks([int(min(lead)), 0, int(max(lead))])

    start, end = find_free_space_between_peaks(all_peaks)

    if end > 0:
        if end != lead_length:
            end -= safety_dist

        if start != 0:
            start += safety_dist

        # axs[12].plot(range(lead_length), np.zeros(lead_length), 'k', [start, end], [0, 0], 'bx', np.arange(start=start, stop=end), np.zeros(end-start), 'b')
        # plt.tight_layout()

        if end != lead_length or start != 0:
            logging.info('Record {} was cropped. New bounds: {}-{}'.format(recid, start, end))
            # plt.savefig('/home/nils/Desktop/crop/cropped/{}.png'.format(recid))
        else:
            logging.info('Record {} was not cropped.'.format(recid))
            # plt.savefig('/home/nils/Desktop/crop/not_cropped/{}.png'.format(recid))

    else:
        logging.warning('Record "{}" could not be cropped due to too many artefacts! You must exclude this record from dataprocessing!'.format(recid))
        # plt.tight_layout()
        # plt.savefig('/home/nils/Desktop/crop/crop_not_possible/{}.png'.format(recid))

    # plt.close()

    return int(start), int(end)


def create_crops_based_on_dataset(crop_id, dataset_id):
    records = load_dataset(dataset_id)

    crops = {}

    for recid in records.keys():
        ecg = records[recid]['ecg_raw']
        lead_length = len(ecg[list(ecg.keys())[0]])
        crop_start, crop_end = find_artefacts_and_crop_ecg(ecg, recid)

        if crop_start != 0 or crop_end != lead_length:
            crops[recid] = {'start': crop_start, 'end': crop_end}

    save_crops(crops, crop_id)
