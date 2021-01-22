# Experiment Environment for Definition, Training and Evaluation of ECG-Based AI Models

The use of artificial intelligence (AI) for analysis of electrocardiogram (ECG) data has recently gained much interest in the AI and medical communities. The discussed models have shown to be able to deliver high diagnostic sensitivity and specificity for detection of various cardiac diseases including rhythm disorders and ischemic events. However, the experiments leading to these results are often difficult to reproduce outside of the original experimental setup and researchers that want to externally validate such results or use them as starting points for new experiments are forced to develop their own models from scratch. We therefore propose a software environment that enables to build, train and evaluate AI models for ECG classification in a reproducible manner and offers sharing of experiment configurations among researchers. The environment further provides simple connection of publicly available data sources of validated ECG recordings. It offers various validation techniques such as bootstrapping and cross-validation. A proof of concept is given for a deep learning model consisting of a convolutional neural network for the classification of acute myocardial infarction based on ECG data.

Authors: Nils Gumpfer, Joshua Prim, Dimitri Grün, Jennifer Hannig, Till Keller, Michael Guckert

AIME 2021, Springer Lecture Notes in Computer Science, https://doi.org/10.101010/9999999


## Setup

For usage of this environment, you require a linux-based server with enough memory (>64GB) and optional GPUs. The environment supports Ubuntu >= 16.04 and can be used with NVIDIA-based GPUs, e.g. on a DGX server.
To install all required software and python packages, run `sudo bash install_requirements.sh`. To download the PTB-XL database as used in the example experiment `ptbxl_proof_of_concept_2_class.ini`, run `bash download_ptbxl.sh`.

## Concept

In the experiment environment, there exists one central configuration for each experiment. Through this configuration (`.ini`) file, all dependent processing steps are controlled. This includes:

- Data preprocessing
- Data splitting into training, validation and testing records
- Experiment conduction
- Experiment evaluation

Not all process steps require all parameters defined in the experiment configuration file, but all source their parameters from the very same file. 

The experiments rely on datasets. These are the result of the data preprocessing step. For the data preprocessing, records from a so-called snapshot (e.g. from PTB-XL) are loaded, filtered, cleansed, transformed, and finally saved as a ready-to-use dataset. This dataset is not yet split into training and validation records. This is done at runtime through so-called split-files. Split files are generated based on the records of a dataset and the validation type defined in the experiment configuration. An experiment is then conducted on the loaded data from the dataset, which is split into training, validation and optionally testing records at runtime. An experiment produces metrics at each epoch (which metrics, is defined in the configuration). After an experiment is finished, these metrics files are loaded and summarized in the evaluation step.

Find the experiment parameters for all steps described above in the next section.

## Experiment parameters

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
data | snapshot_id | yes | String | The ID of the snapshot that should be used for data extraction during preprocessing. | 
data | record_ids_excluded | no | String, comma-separated list (no space) | List of record-IDs that should be excluded during data-preprocessing (e.g. because of wrong/missing data) |
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
hyperparameters_general | batch_size | no | ??? | ??? |
hyperparameters_ecgmodel_flex | number_filters_conv | no | Integer, comma-separated list (no space) | List of number of conv filters per layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | size_kernel_conv | no | Integer, comma-separated list (no space) | List of kernel sizes per conv layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | size_kernel_pool | no | Integer, comma-separated list (no space) | List of kernel sizes per pooling layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | stride_conv | no | Integer, comma-separated list (no space) | List of stride lengths per conv layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | stride_pool | no | Integer, comma-separated list (no space)| List of stride lengths per pooling layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | padding_conv | no | “valid”, “same”, comma-separated list (no space) | List of padding types per conv layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | maxpooling_conv | no | Boolean, comma-separated list (no space) | List of booleans to specify maxpooling after conv layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | batchnorm_conv | no | Boolean, comma-separated list (no space) | List of booleans to specify batchnormalization after conv layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | dropout_rate_conv | no | Double, comma-separated list (no space) | List of dropout rates to specify dropout after conv layer (only for conv part of model). Zero means no dropout.|
hyperparameters_ecgmodel_flex | activation_function_conv | no | “relu”, “elu”, “leaky_relu”, “softmax”, comma-separated list (no space) | List of activation functions per layer (only for conv part of model)|
hyperparameters_ecgmodel_flex | transition_conv_dense | no | “flatten”, “GAP” | Transition type from conv to dense part of model |
hyperparameters_ecgmodel_flex | number_neurons_dense | no | Integer, comma-separated list (no space) | List of number of neurons per layer (only for dense part of model)|
hyperparameters_ecgmodel_flex | batchnorm_dense | no | Boolean, comma-separated list (no space) | List of booleans to specify batchnormalization after dense layer (only for dense part of model)|
hyperparameters_ecgmodel_flex | dropout_rate_dense | no | Double, comma-separated list (no space) |  List of dropout rates to specify dropout after dense layer (only for dense part of model, except last layer). Zero means no dropout.|
hyperparameters_ecgmodel_flex | activation_function_dense | no | “relu”, “elu”, “leaky_relu”, “softmax”, comma-separated list (no space) | List of activation functions per layer (only for dense part of model)|
evaluation | metrics | no | String, comma-separated list (no space) | Metrics to calculate during training and in the evaluation step. If no metrics are given, senstivity, specificity and AUC are calculated by default |
evaluation | calculation_methods | no | String, comma-separated list (no space) | Calclulation methods of metrics. Possible values: “sample_level”, “subsample_level”. For tensorboard, only the first method will be used |
evaluation | class_names | yes | String, comma-separated list (no space) | Class names of output classes that should be considered for evaluation |
evaluation | target_metric | yes | String | The target metric for choosing the “best” model(s) from all epochs |
evaluation | tensorboard_subdir | yes | String | The subdir for tensorboard (you should create a new directory when tensorboard becomes slow; delete the old one if experiments are closed.) |
evaluation | sensitivity_threshold | yes | Float between 0 and 1 | The threshold for epoch/model selection. All models weaker than this value will not be considered for final metric calculation |
evaluation | specificity_threshold | yes | Float between 0 and 1 | The threshold for epoch/model selection. All models weaker than this value will not be considered for final metric calculation |
evaluation | save_raw_results | no | Boolean | The unfiltered result list containing all epochs can be saved to disk. Consider, that this file will be very large and requires much diskspace! |

## Process steps
### Data preprocessing

To run data preprocessing for an experiment, switch to directory `/runner/preprocessing_runner` and run `python3 preprocessing_runner.py -e $EXPERIMENT_ID$`, for example `python3 preprocessing_runner.py -e ptbxl_proof_of_concept_2_class`

### Data splitting

To run data splitting for an experiment, switch to directory `/runner/splitting_runner` and run `python3 splitting_runner.py -e $EXPERIMENT_ID$`, for example `python3 splitting_runner.py -e ptbxl_proof_of_concept_2_class`

### Experiment conduction

To run an experiment, switch to directory `/runner/experiment_runner` and run `python3 experiment_runner.py -e $EXPERIMENT_ID$`, for example `python3 experiment_runner.py -e ptbxl_proof_of_concept_2_class`. Experiment evaluation is performed automatically afterwards.

### Experiment evaluation

To manually (re-)evaluate an experiment, switch to directory `/runner/evaluation_runner` and run `python3 evaluation_runner.py -e $EXPERIMENT_ID$`, for example `python3 evaluation_runner.py -e ptbxl_proof_of_concept_2_class`