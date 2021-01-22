from utils.file.file import save_dict_as_json

# TODO: Here, load your clinical parameters and perform the below operation in a loop for all your records
clinical_parameters = {'age': 21, 'sex': 'f', 'MI': 1}
save_dict_as_json(clinical_parameters, 'data/custom/snapshots/v1/RECORD123.json')
