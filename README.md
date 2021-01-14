# How to work with this experiment environment on Yamata
## General steps

#### Environment
We use an NVIDIA DGX-1 computing environment provided and maintained by MNI in Gießen, codename 'YAMATA'. Responsible for this server is Prof. Dr. Andreas Dominik. Contact person in his workgroup is Eric Hartmann.

#### Access
Ask Eric Hartmann for access (eric.hartmann@mni.thm.de) and a port number you can use for tensorboard.

#### Login
Login to VPN or directly use the THM network to be able to connect to the server. Connect via SSH using your THM account and standard (email) password
`ssh $THMUSER$@bioinf-yamata.mni.thm.de`, for example `ssh ngmp33@bioinf-yamata.mni.thm.de`.

#### Docker container setup

##### Build and start container
Run `docker run -it -d -v /raid/$THMUSER$:/data --gpus all -p $OPENEDPORT$:6006 --name $CONTAINERNAME$ $IMAGENAME$:$IMAGETAG$ bash` to create a new container based on an image, for example
`docker run -it -d -v /raid/ngmp33:/data --gpus all -p 7900:6006 --name ngmp33_exp1 ngmp33_exp:v1.0 bash`.

Our current stable image is `ngmp33_exp:v1.2`.

##### Connect to container
Run `docker exec -it $CONTAINERNAME$ bash` to connect to your container, for example `docker exec -it ngmp33_exp1 bash`.

##### Setup environment
In the `/home` directory inside the container, run the shell script `environment_setup.sh` via `bash environment_setup.sh`.

1. You will be prompted to login to the gitlab repository

2. You will be prompted to login to google cloud

3. Data from PTB-XL will be downloaded (see https://physionet.org/content/ptb-xl/)

#### Disconnect from a container
Run `exit` inside the container to come back to the yamata console.

##### Screens

To be able to run multiple parallel sessions / processes on the server, we use screens (virtual consoles). You can attach to and detach from such a virtual console.

To create a screen, run `screen` and accept the prompt with some key.

In the screen, you can work like on the normal console. To detach from the screen, press `CTRL + A`, followed by `D`

To attach again to a screen, run `screen -r`. This won't work when you opened more than one screen.

To list all opened screens, run `screen -ls`. To attach to a specific screen, run `screen -r $SCREEN_ID$`. You can accelerate this step by typing the first numbers of the ID and then hitting `TAB`.

To close a screen, run `exit` when you are attached to a screen.

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
data | crop_id | no | String | The ID of the crop marker file located in data/crops/ (without filetype). The crop marker file generation done running the crop_runner in the runner package. |
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
hyperparameters_ecgmodel | ecgmodel_initializer_conv | no | String | Convolutional weight initializer of ECG model |
hyperparameters_ecgmodel | ecgmodel_initializer_dense | no | String | Dense layer weight initializer of ECG model |
hyperparameters_ecgmodel | ecgmodel_number_layers_conv | no | Integer | Number of convolutional layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_number_filters_conv | no | Integer | Number of convolutional filters of ECG model |
hyperparameters_ecgmodel | ecgmodel_number_layers_dense | no | Integer | Number of dense layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_number_neurons_dense | no | Integer | Number of neurons per dense layer of ECG model |
hyperparameters_ecgmodel | ecgmodel_size_kernel_conv | no | Integer | Size of convolutional kernel of ECG model |
hyperparameters_ecgmodel | ecgmodel_size_kernel_pool | no | Integer | Size of pooling kernel of ECG model |
hyperparameters_ecgmodel | ecgmodel_stride_conv | no | Integer | Length of convolutional stride of ECG model |
hyperparameters_ecgmodel | ecgmodel_stride_pool | no | Integer | Length of pooling stride of ECG model |
hyperparameters_ecgmodel | ecgmodel_padding_conv | no | “valid”, “causal”, “same” | Type of padding of convolutional layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_dropout_conv | no | Boolean | If dropout should be used between convolutional layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_dropout_dense | no | Boolean | If dropout should be used between dense layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_dropout_rate_conv | no | Float | Dropout rate that should be used between convolutional layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_dropout_rate_dense | no | Float | Dropout rate that should be used between dense layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_activation_function_conv | no | String | Activation function of convolutional layers of ECG model |
hyperparameters_ecgmodel | ecgmodel_activation_function_dense | no | String | Activation function of dense layers of ECG model |
hyperparameters_clinicalparametermodel | clinicalparametermodel_initializer_dense | no | String | Dense layer weight initializer of clinical-parameter-model |
hyperparameters_clinicalparametermodel | clinicalparametermodel_dropout_dense | no | Boolean | If dropout should be used between dense layers of clinical-parameter-model |
hyperparameters_clinicalparametermodel | clinicalparametermodel_dropout_rate_dense | no | Float | Dropout rate that should be used between dense layers of clinical-parameter-model |
hyperparameters_clinicalparametermodel | clinicalparametermodel_activation_function_dense | no | String | Activation function of dense layers of clinical-parameter-model |
hyperparameters_clinicalparametermodel | clinicalparametermodel_number_neurons_dense | no | Integer | Number of neurons per dense layer of clinical-parameter-model |
hyperparameters_clinicalparametermodel | clinicalparametermodel_number_layers_dense | no | Integer | Number of dense layers of clinical-parameter-model |
hyperparameters_combinationmodel | combinationmodel_initializer_dense | no | String | Dense layer weight initializer of combination-model |
hyperparameters_combinationmodel | combinationmodel_dropout_dense | no | Boolean | If dropout should be used between dense layers of combination-model |
hyperparameters_combinationmodel | combinationmodel_dropout_rate_dense | no | Float | Dropout rate that should be used between dense layers of combination-model |
hyperparameters_combinationmodel | combinationmodel_activation_function_dense | no | String | Activation function of dense layers of combination-model |
hyperparameters_combinationmodel | combinationmodel_number_neurons_dense | no | Integer | Number of neurons per dense layer of combination-model |
hyperparameters_combinationmodel | combinationmodel_number_layers_dense | no | Integer | Number of dense layers of combination-model |
explanation | low_level_method | no | String | Low-level explanation method (iNNvestigate analyzer ID, https://github.com/albermax/innvestigate) → still experimental |
explanation | high_level_method | no | String | High-level explanation method (e.g. heatmap clustering, etc.) → still experimental |
explanation | xai_class_name | no | String | Class name of output class that should be considered for explanation |
transfer_learning | load_model_experiment_id | no | String | (Sub)experiment-ID of the model that should be loaded for transfer learning |
transfer_learning | load_model_epoch | no | Integer | Epoch number of the model that should be loaded for transfer learning |
transfer_learning | remove_last_layers | no | Integer | Number of last model layers to remove (usually the dense classifier layers) |

#### Data preprocessing

To run data preprocessing for an experiment, switch to directory `/runner/preprocessing_runner` and run `python3 preprocessing_runner.py -e $EXPERIMENT_ID$`, for example `python3 preprocessing_runner.py -e 2020-07_my_first_experiment`

#### Data splitting

To run data splitting for an experiment, switch to directory `/runner/splitting_runner` and run `python3 splitting_runner.py -e $EXPERIMENT_ID$`, for example `python3 splitting_runner.py -e 2020-07_my_first_experiment`

#### ECG cropping

To run ECG cropping for an experiment, switch to directory `/runner/splitting_runner` and run `python3 cropping_runner.py -e $EXPERIMENT_ID$`, for example `python3 cropping_runner.py -e 2020-07_my_first_experiment`

#### Experiment conduction

To run an experiment, switch to directory `/runner/experiment_runner` and run `python3 experiment_runner.py -e $EXPERIMENT_ID$`, for example `python3 experiment_runner.py -e 2020-07_my_first_experiment`

#### Experiment evaluation

To evaluate an experiment, switch to directory `/runner/evaluation_runner` and run `python3 evaluation_runner.py -e $EXPERIMENT_ID$`, for example `python3 evaluation_runner.py -e 2020-07_my_first_experiment`


### Web Interface

#### Tensorboard

To start tensorboard, open a new screen and run `tensorboard --logdir=/data/ecg-dl-experiment-environment/tensorboard/$SUBDIR$ --bind_all`. You can reach it via `https://bioinf-yamata.mni.thm.de:$OPENEDPORT$`, for example `https://bioinf-yamata.mni.thm.de:7900`. 

**Important notice:** For each experiment, there is a mandatory parameter in the config file called `tensorboard_subdir`, which specifies where the tensorboard files are logged to. This is due to the fact that tensorboard becomes very slow if a large amount of logs has to be managed. For this reason, the logs are separated into subdirs.

#### Experiment overview

To get an overview of all experiments where you can also access the evaluation results, open a new screen and run `python3 manage.py runserver 0:6006`. You can reach it via `https://bioinf-yamata.mni.thm.de:$OPENEDPORT$`, for example `https://bioinf-yamata.mni.thm.de:7900`. 

_Hint:_ Keep in mind that the experiment overview has to run on a different port than tensorboard. Otherwise, open two separate ports.

#### File system browser

To access the file system of your docker container, go to `https://bioinf-yamata.mni.thm.de:$OPENEDPORT$/filesystembrowser`. Make sure to run the steps descibed in "Experiment overview" above to start the web service.