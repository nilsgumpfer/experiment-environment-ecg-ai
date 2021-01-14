import importlib
import logging

from runner.abstract_runner import AbstractRunner
from utils.file.file import parse_experiment_config
from utils.misc.gpu import set_visible_gpu
from utils.misc.logger import create_log_directory, initialize_logger
from utils.misc.text import convert_underscored_to_camelcaps


class SplitRunner(AbstractRunner):
    def __init__(self, exp_id=None, default_yes=None):
        super().__init__(exp_id, default_yes)

        self.params = parse_experiment_config(self.experiment_id)

        set_visible_gpu(self.params['gpu_id'])

        initialize_logger(logdir=None, loglevel=self.params['loglevel'], only_console=True)
        logging.debug('Reading the parameters was successful.')

    def run(self):
        path_to_splitter = 'splitting.' + self.params['splitter_id']

        module = importlib.import_module(path_to_splitter)
        classname = convert_underscored_to_camelcaps(self.params['splitter_id'])
        my_class = getattr(module, classname)
        my_object = my_class(self.params)

        my_object.perform_splitting()


if __name__ == '__main__':
    ppr = SplitRunner()
    try:
        ppr.run()
    except Exception as e:
        logging.error(str(e))
        raise Exception(e.args)
