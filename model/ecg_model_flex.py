import logging

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout, Softmax, \
    BatchNormalization, Flatten, LeakyReLU, ReLU, ELU

from dataloading.basic_dataloader import BasicDataloader
from model.abstract_model import AbstractModel
from utils.data.data import derive_binary_one_hot_classes_for_list_of_labels
from utils.experiments.callbacks import CustomCallbackV1
from utils.experiments.training import get_optimizer, derive_batch_size


class EcgModelFlex(AbstractModel):
    def __init__(self, params):
        super().__init__(params)
        self.train_clinical_parameters = None
        self.train_diagnosis = None
        self.train_ecg_raw = None
        self.train_metadata = None
        self.validation_clinical_parameters = None
        self.validation_diagnosis = None
        self.validation_ecg_raw = None
        self.validation_metadata = None
        self.output_layer_neurons = None
        self.input_shape_ecgmodel = None
        self.metrics = None
        self.calculation_methods = None
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        self.y_classes = None
        self.train_record_ids = None
        self.validation_record_ids = None
        self.tensorboard_logdir = self.params['tensorboard_logdir']
        self.experiment_logdir = self.params['experiment_logdir']
        self.dataloader = BasicDataloader(self.params)

    def load_data(self):
        self.train_record_ids, self.train_metadata, self.train_diagnosis, self.train_clinical_parameters, self.train_ecg_raw, self.validation_record_ids, self.validation_metadata, self.validation_diagnosis, self.validation_clinical_parameters, self.validation_ecg_raw = self.dataloader.load_data()

        self.output_layer_neurons = len(self.train_diagnosis[0])
        self.input_shape_ecgmodel = (self.params['subsampling_window_size'], len(self.params['leads_to_use']))

        self.metrics = self.params['metrics']
        self.calculation_methods = self.params['calculation_methods']

        self.x_train = [np.asarray(self.train_ecg_raw)]
        self.x_val = [np.asarray(self.validation_ecg_raw)]
        self.y_train = np.asarray(self.train_diagnosis)
        self.y_val = np.asarray(self.validation_diagnosis)

        self.y_classes = derive_binary_one_hot_classes_for_list_of_labels(self.params['clinical_parameters_outputs'])

        logging.info('Dataloading completed.')

    def create_model(self):
        # Build model
        model = Sequential()

        # Conv
        for i, _conv_number_filters, _conv_size_kernel, _conv_stride, _conv_padding, _conv_activation, _conv_batchnorm, _conv_dropout, _maxpool, _maxpool_stride, _maxpool_size_kernel in zip(range(len(self.params['ecgmodel_number_filters_conv'])), self.params['ecgmodel_number_filters_conv'], self.params['ecgmodel_size_kernel_conv'], self.params['ecgmodel_stride_conv'], self.params['ecgmodel_padding_conv'], self.params['ecgmodel_activation_function_conv'], self.params['ecgmodel_batchnorm_conv'], self.params['ecgmodel_dropout_rate_conv'], self.params['ecgmodel_maxpooling_conv'], self.params['ecgmodel_stride_pool'], self.params['ecgmodel_size_kernel_pool']):
            if i == 0:
                model.add(Conv1D(kernel_size=_conv_size_kernel,
                                 filters=_conv_number_filters,
                                 padding=_conv_padding,
                                 input_shape=self.input_shape_ecgmodel))
            else:
                model.add(Conv1D(kernel_size=_conv_size_kernel,
                                 filters=_conv_number_filters,
                                 padding=_conv_padding))

            print('Conv', _conv_number_filters, _conv_size_kernel, _conv_padding)

            # Batch normalization
            if _conv_batchnorm:
                model.add(BatchNormalization())
                print('BatchNorm')

            # Activation
            if _conv_activation == 'elu':
                model.add(ELU())
                print('ELU')
            elif _conv_activation == 'relu':
                model.add(ReLU())
                print('ReLU')
            elif _conv_activation == 'leaky_relu':
                model.add(LeakyReLU())
                print('LeakyReLU')

            # Max pooling
            if _maxpool:
                model.add(MaxPool1D(strides=_maxpool_stride, pool_size=_maxpool_size_kernel))
                print('Max pooling')

            # Dropout
            if _conv_dropout > 0:
                model.add(Dropout(_conv_dropout))
                print('Dropout', _conv_dropout)

        # Transition
        if self.params['ecgmodel_transition_conv_dense'] == 'GAP':
            model.add(GlobalAveragePooling1D())
            print('GAP')
        elif self.params['ecgmodel_transition_conv_dense'] == 'flatten':
            model.add(Flatten())
            print('Flatten')

        # Dense
        for _dense_width_layer, _dense_activation, _dense_dropout, _dense_batchnorm in zip(self.params['ecgmodel_number_neurons_dense'], self.params['ecgmodel_activation_function_dense'], self.params['ecgmodel_dropout_rate_dense'], self.params['ecgmodel_batchnorm_dense']):
            model.add(Dense(units=_dense_width_layer))
            print('Dense', _dense_width_layer)

            # Batch normalization
            if _dense_batchnorm:
                model.add(BatchNormalization())
                print('BatchNorm')

            # Activation
            if _dense_activation == 'elu':
                model.add(ELU())
                print('ELU')
            elif _dense_activation == 'relu':
                model.add(ReLU())
                print('ReLU')
            elif _dense_activation == 'leaky_relu':
                model.add(LeakyReLU())
                print('LeakyReLU')
            elif _dense_activation == 'softmax':
                model.add(Softmax())
                print('Softmax')

            # Dropout
            if _dense_dropout > 0:
                model.add(Dropout(_dense_dropout))
                print('Dropout', _dense_dropout)

        self.model = model

    def train_model(self):
        tensorboard_callback = TensorBoard(log_dir=self.tensorboard_logdir)

        custom_callback = CustomCallbackV1(x_val=self.x_val,
                                           y_val=self.y_val,
                                           y_classes=self.y_classes,
                                           record_ids_val=self.validation_record_ids,
                                           metrics=self.metrics,
                                           calculation_methods=self.calculation_methods,
                                           experiment=self.params['experiment_id'],
                                           experiment_logdir=self.experiment_logdir,
                                           tensorboard_logdir=self.tensorboard_logdir,
                                           class_names_to_log=self.params['class_names'],
                                           metric_thresholds={'sensitivity': self.params['sensitivity_threshold'], 'specificity': self.params['specificity_threshold']})

        optimizer = get_optimizer(name=self.params['optimizer'],
                                  learning_rate=self.params['learning_rate'],
                                  learning_rate_decay=self.params['learning_rate_decay'])

        self.model.compile(optimizer=optimizer,
                           loss=self.params['loss_function'])

        self.model.fit(x=self.x_train,
                       y=self.y_train,
                       batch_size=derive_batch_size(self.params['batch_size'], len(self.train_record_ids)),
                       epochs=self.params['number_epochs'],
                       shuffle=self.params['shuffle'],
                       validation_data=(self.x_val, self.y_val),
                       callbacks=[tensorboard_callback, custom_callback])
