import time

from utils.api.snomed_api import getConceptById
from utils.data.data import load_ecgs_from_georgia, load_ecgs_from_georgia_grouped
from utils.file.file import pickle_data, unpickle_data
import numpy as np
import pandas as pd

# x = load_ecgs_from_georgia('WFDB_Ga', snapshot_directory='./data/georgia/snapshots', leads_to_use=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
#
# for item in x:
#     x[item].pop('leads')
#
# pickle_data(x, 'data/georgiadb.pickled')
#
# records = unpickle_data('data/georgiadb.pickled')
#
# labels = [x['labels'] for x in records.values()]
#
# all_labels = []
#
# for l in labels:
#     all_labels += l
#
# d = []
#
# for c in sorted(list(set(all_labels))):
#     d.append({'label': c, 'text': getConceptById(c), 'count': all_labels.count(c)})
#
# df = pd.DataFrame(d)
# df.to_excel('data/snomed_labels.xls', index=False)
# df.to_csv('data/snomed_labels.csv', index=False)

# for item in x:
#     if '426783006' not in x[item]['labels']:
#         x[item].pop('leads')
#
# print(len(x))
# pickle_data(x, 'data/georgiadb_426783006.pickled')
# from utils.viz.ecg import plot_ecg
#
# records = unpickle_data('data/georgiadb_426783006.pickled')
#
# for record_id in records:
#     if 'leads' in records[record_id].keys():
#         plot_ecg(records[record_id], subdict='leads', title='{}{}'.format(record_id, records[record_id]['labels']), save_to='data/plots/{}.jpg'.format(record_id))
#
# print(records)

records = load_ecgs_from_georgia_grouped('WFDB_Ga', snapshot_directory='./data/georgia/snapshots', leads_to_use=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

rows = []
groups = []

for recid in records:
    groups.append(records[recid]['groups'][0])
    rows.append({'record_id': recid, 'groups': records[recid]['groups']})

# df = pd.DataFrame(rows)
# df.to_excel('data/georgia_records_grouped.xls', index=False)

print('healthy', groups.count('healthy'))
print('ischemia', groups.count('ischemia'))
print('other', groups.count('other'))

