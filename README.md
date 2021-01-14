# How to work with this experiment environment
## General steps

TODO: Use Ubuntu, Nvidia cards, etc.

## Working with the enviromnent

### Concept

In the experiment environment, there exists one central configuration for each experiment. Through this configuration (`.ini`) file, all dependent processing steps are controlled. This includes:

- Data preprocessing
- Data splitting into training, validation and testing records
- ECG cropping
- Experiment conduction
- Experiment evaluation
- Model and prediction explanation

Not all process steps require all parameters defined in the experiment configuration file, but all source their parameters from the very same file. 

The experiments rely on datasets. These are the result of the data preprocessing step. For the data preprocessing, records from a so-called snapshot (e.g. from Redcap, PTB-XL) are loaded, filtered, cleansed, transformed, and finally saved as a ready-to-use dataset. This dataset is not yet split into training and validation records. This is done at runtime through so-called split-files. Split files are generated based on the records of a dataset and the validation type defined in the experiment configuration. An experiment is then conducted on the loaded data from the dataset, which is split into training, validation and optionally testing records at runtime. An experiment produces metrics at each epoch (which metrics, is defined in the configuration). After an experiment is finished, these metrics files are loaded and summarized in the evaluation step. Explanations for predictions and the model can be generated in the explanation step (still experimental).

Find the experiment parameters for all steps described above in the next section.

### Experiment parameters

Group | Parameter | Required | Possible Values | Description |
|---|---|---|---|---|
general | experiment_series | yes | String | Series of experiments, which will be used to group experiments |
general | question | yes | String | The question you woul like to answer with the experiment |
general | hypothesis | yes | String | Your hypothesis for this experiment |
general | remarks | no | String | Any remarks that could be necessary to understand this experiment |
environment | model_id | yes | String | ID of the model class that will be used to build the model and perform training |
environment | random_seed | no | Integer | Random seed that will be used throughout all steps where random values are used. This enables full experiment reproducibility. |
environment | preprocessor_id | yes | String | ID of the preprocessor class that will be used to perform data preprocessing |
environment | splitter_id | yes | String | ID of the splitter class that will be used to generate data splits (training, validation, test) |
environment | evaluator_id | yes | String | ID of the evaluator class that will be used to generate evaluations of the experiment such as excel files and PDFs |
environment | explanator_id | no | String | ID of the explanator class that will be used to generate explanations such as heatmaps (currently in experimental state!) |
environment | gpu_id | yes | 6, 7 | ID of the GPU that should be used for training. We are allowed to use GPUs 6 and 7 only! |
environment | loglevel | yes | “DEBUG”, “INFO”, “WARNING”, “ERROR” | Loglevel of the logger |
data | dataset_id | yes | String | The ID of the dataset that will be generated during preprocessing and used during training and all further steps. |
data | split_id | yes | String | The ID of the split that will be used to generate the sub-splits during splitting and that will be used to split the data of the dataset during training. Data is split during training, not during preprocessing! |
data | leads_to_use | yes | String, comma-separated list (no space) | List of Lead-IDs that should be used by the model. All other leads are discared during preprocessing. The dataset will only contain the selected leads. The model will be structured based on the dataset, not based on this parameter during training! |
data | clinical_parameters_outputs | yes | String, comma-separated list (no space) | The clinical parameters to use as outputs for the model. Only binary parameters can be used currently. Each parameter will be one-hot-encoded with a _TRUE and _FALSE suffix. |
data | subsampling_factor | yes | Integer | The factor that should be used for subsampling. Subsampling is not applied during preprocessing, but during training! |
data | subsampling_window_size | yes | Integer | The window size that should be used for subsampling. Subsampling is not applied during preprocessing, but during training! |
data | clinical_parameters_inputs | no | String, comma-separated list (no space) | The clinical parameters to use as inputs for the model. If this parameter is not provided for preprocessing, no input parameters will be available during training. |
data | ecg_variants | yes | String, comma-separated list (no space) | The variants of the ECG the model should be able to use later during training. Currently, only ecg_raw is implemented. In future, als ecg_delta will be available |
data | snapshot_id | yes | String, comma-separated list (no space) | The IDs of the snapshots that should be used for data extraction during preprocessing. Multiple snapshot names are only allowed for Kerckhoff data. The record-IDs have to be unique, to prevent overlapping snapshots. For PTB-XL data, only the first value in the list will be used. |
data | record_ids_excluded | no | String, comma-separated list (no space) | List of record-IDs that should be excluded during data-preprocessing (e.g. because of wrong/missing data) |
data | source_id | no | String | Currently, this parameter is not in use. Later, it could serve for further data-type differentiation |
data | metadata_id | yes | String | The ID of the metadata file located in data/metadata/ (without filetype) |
data | stratification_variable | yes | String | The name of the output variable that should be used for stratification of the train/validation/test splits  |
data | ratio_split | no | Float between 0 and 1 | Split ratio for simple splits, required for splitting, when validation_type “single” is used |
data | ratio_test | no | Float between 0 and 1 | Split ratio for test data holdout |
hyperparameters_general | number_epochs | yes | Integer | The number of epochs each training will last |
hyperparameters_general | optimizer | yes | String | The model optimizer, https://www.tensorflow.org/api_docs/python/tf/keras/optimizers |
hyperparameters_general | learning_rate | yes | Float | Optimizer learning rate |
hyperparameters_general | learning_rate_decay | no | Float | Decay of learning rate of optimizer |
hyperparameters_general | shuffle | yes | Boolean | If records should be suffled before each |
hyperparameters_general | loss_function | yes | String | The loss function, https://www.tensorflow.org/api_docs/python/tf/keras/losses |
hyperparameters_general | number_training_repetitions | no | Integer | The number of times the training (all epochs) will be repeated (each repetition uses a different random seed for weight initialization) |
hyperparameters_general | validation_type | no | “cross_validation”, “bootstrapping”, “single” | If no validation type is given, the validation_type “single” is chosen, which uses the best model of the training |
hyperparameters_general | folds_cross_validation | no | Integer | Only required when validation_type “cross_validation” is used. Number of folds (k) |
hyperparameters_general | bootstrapping_n | no | Integer | Only required when validation_type “bootstrapping” is used. Number of draws from data population (n) |
evaluation | metrics | no | String, comma-separated list (no space) | Metrics to calculate during training and in the evaluation step. If no metrics are given, senstivity, specificity and AUC are calculated by default |
evaluation | calculation_methods | no | String, comma-separated list (no space) | Calclulation methods of metrics. Possible values: “sample_level”, “subsample_level”. For tensorboard, only the first method will be used |
evaluation | class_names | yes | String, comma-separated list (no space) | Class names of output classes that should be considered for evaluation |
evaluation | target_metric | yes | String | The target metric for choosing the “best” model(s) from all epochs |
evaluation | recipients_emails | no | String, comma-separated list (no space) | Evaluation results will be broadcasted to these recipients via eMail  |
evaluation | tensorboard_subdir | yes | String | The subdir for tensorboard (you should create a new directory when tensorboard becomes slow; delete the old one if experiments are closed.) |
evaluation | sensitivity_threshold | yes | Float between 0 and 1 | The threshold for epoch/model selection. All models weaker than this value will not be considered for final metric calculation |
evaluation | specificity_threshold | yes | Float between 0 and 1 | The threshold for epoch/model selection. All models weaker than this value will not be considered for final metric calculation |
evaluation | save_raw_results | no | Boolean | The unfiltered result list containing all epochs can be saved to disk. Consider, that this file will be very large and requires much diskspace! |

TODO: describe hyperparameters for flex model


#### Data preprocessing

To run data preprocessing for an experiment, switch to directory `/runner/preprocessing_runner` and run `python3 preprocessing_runner.py -e $EXPERIMENT_ID$`, for example `python3 preprocessing_runner.py -e 2020-07_my_first_experiment`

#### Data splitting

To run data splitting for an experiment, switch to directory `/runner/splitting_runner` and run `python3 splitting_runner.py -e $EXPERIMENT_ID$`, for example `python3 splitting_runner.py -e 2020-07_my_first_experiment`

#### Experiment conduction

To run an experiment, switch to directory `/runner/experiment_runner` and run `python3 experiment_runner.py -e $EXPERIMENT_ID$`, for example `python3 experiment_runner.py -e 2020-07_my_first_experiment`

#### Experiment evaluation

To evaluate an experiment, switch to directory `/runner/evaluation_runner` and run `python3 evaluation_runner.py -e $EXPERIMENT_ID$`, for example `python3 evaluation_runner.py -e 2020-07_my_first_experiment`