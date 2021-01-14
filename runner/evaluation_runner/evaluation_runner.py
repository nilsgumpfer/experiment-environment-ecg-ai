import importlib
import logging

from tensorflow.python.keras.backend import clear_session

from runner.abstract_runner import AbstractRunner
from utils.experiments.training import enable_reproducibility
from utils.file.file import logdir_exists, parse_experiment_config
from utils.misc.gpu import set_visible_gpu
from utils.misc.logger import initialize_logger
from utils.misc.text import convert_underscored_to_camelcaps


class EvaluationRunner(AbstractRunner):
    def __init__(self, exp_id=None, default_yes=None):
        super().__init__(exp_id, default_yes)

        logdir, exists = logdir_exists(self.experiment_id, category='experiments')

        if exists is False:
            raise Exception('No logdir present for experiment "{}". Cannot perform evaluation. Aborting.'.format(
                self.experiment_id))

        self.params = parse_experiment_config(self.experiment_id)

        set_visible_gpu(self.params['gpu_id'])

        clear_session()
        enable_reproducibility(self.params['random_seed'])

        initialize_logger(logdir, loglevel=self.params['loglevel'], append=True)
        logging.debug('Reading the parameters was successful.')

    def run(self):
        path_to_evaluator = 'evaluation.' + self.params['evaluator_id']

        module = importlib.import_module(path_to_evaluator)
        classname = convert_underscored_to_camelcaps(self.params['evaluator_id'])
        my_class = getattr(module, classname)
        my_object = my_class(self.params, self.experiment_id)

        my_object.perform_evaluation()


if __name__ == '__main__':
    evr = EvaluationRunner()
    try:
        evr.run()
    except Exception as e:
        logging.error(str(e))
        raise Exception(e.args)
