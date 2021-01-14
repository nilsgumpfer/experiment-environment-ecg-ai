import importlib
import logging

from tensorflow.python.keras.backend import clear_session

from runner.abstract_runner import AbstractRunner
from utils.experiments.training import enable_reproducibility
from utils.experiments.validation import single_repeated, cross_validation, bootstrapping, cross_validation_repeated, \
    bootstrapping_repeated
from utils.file.file import cleanup_directory, parse_experiment_config
from utils.misc.gpu import set_visible_gpu
from utils.misc.logger import create_log_directory, initialize_logger
from utils.misc.text import convert_underscored_to_camelcaps


class ExperimentRunner(AbstractRunner):
    def __init__(self, exp_id=None, default_yes=None):
        super().__init__(exp_id, default_yes)

        logdir = create_log_directory(self.experiment_id, category='experiments', default_yes=self.default_yes)

        self.params = parse_experiment_config(self.experiment_id)

        self.params['experiment_logdir'] = logdir
        self.params['tensorboard_logdir'] = '../../tensorboard/{}/{}'.format(self.params['tensorboard_subdir'], self.params['experiment_id'])
        cleanup_directory(self.params['tensorboard_logdir'])

        self.repeated = self.params['number_training_repetitions'] not in [None, 0]

        set_visible_gpu(self.params['gpu_id'])

        clear_session()
        enable_reproducibility(self.params['random_seed'])

        initialize_logger(logdir, loglevel=self.params['loglevel'])

    def run(self):
        if self.params['validation_type'] in [None, 'single']:
            if self.repeated:
                self.run_single_repeated()
            else:
                self.run_single()
        elif self.params['validation_type'] == 'cross_validation':
            if self.repeated:
                self.run_cross_validation_repeated()
            else:
                self.run_cross_validation()
        elif self.params['validation_type'] == 'bootstrapping':
            if self.repeated:
                self.run_bootstrapping_repeated()
            else:
                self.run_bootstrapping()

    def run_evaluation(self):
        path_to_evaluator = 'evaluation.' + self.params['evaluator_id']

        module = importlib.import_module(path_to_evaluator)
        classname = convert_underscored_to_camelcaps(self.params['evaluator_id'])
        my_class = getattr(module, classname)
        my_object = my_class(self.params, self.experiment_id)

        my_object.perform_evaluation()

    def run_single(self):
        logging.info('Running experiment "{}" on GPU {}'.format(self.experiment_id, self.params['gpu_id']))
        path_to_model = 'model.' + self.params['model_id']

        module = importlib.import_module(path_to_model)
        classname = convert_underscored_to_camelcaps(self.params['model_id'])
        my_class = getattr(module, classname)
        my_object = my_class(self.params)

        logging.info('Loading data...')
        my_object.load_data()

        logging.info('Creating model...')
        my_object.create_model()

        logging.info('Starting training...')
        my_object.train_model()

    def run_single_repeated(self):
        logging.info('Running single repeated experiment "{}"'.format(self.experiment_id))
        single_repeated(self.experiment_id, self.params['number_training_repetitions'], self, random_seed=self.params['random_seed'])

    def run_cross_validation(self):
        logging.info('Running cross validation experiment "{}"'.format(self.experiment_id))
        cross_validation(self.experiment_id, self.params['folds_cross_validation'], self.params['split_id'], self)

    def run_bootstrapping(self):
        logging.info('Running bootstrapping experiment "{}"'.format(self.experiment_id))
        bootstrapping(self.experiment_id, self.params['bootstrapping_n'], self.params['split_id'], self)

    def run_cross_validation_repeated(self):
        logging.info('Running repeated cross validation experiment "{}"'.format(self.experiment_id))
        cross_validation_repeated(self.experiment_id, self.params['folds_cross_validation'], self.params['number_training_repetitions'], self.params['split_id'], self, random_seed=self.params['random_seed'])

    def run_bootstrapping_repeated(self):
        logging.info('Running repeated bootstrapping experiment "{}"'.format(self.experiment_id))
        bootstrapping_repeated(self.experiment_id, self.params['bootstrapping_n'], self.params['number_training_repetitions'], self.params['split_id'], self, random_seed=self.params['random_seed'])

    def run_list_of_experiments(self, experiments):
        for experiment in experiments:
            exr = ExperimentRunner(exp_id=experiment, default_yes=self.default_yes)
            exr.run()


if __name__ == '__main__':
    exr = ExperimentRunner()
    try:
        exr.run()
        exr.run_evaluation()

    except Exception as e:
        logging.error(str(e))
        raise Exception(e.args)
