import logging

from runner.abstract_runner import AbstractRunner
from utils.data.cropping import create_crops_based_on_dataset
from utils.file.file import parse_experiment_config
from utils.misc.gpu import set_visible_gpu
from utils.misc.logger import initialize_logger


class CropRunner(AbstractRunner):
    def __init__(self, exp_id=None, default_yes=None):
        super().__init__(exp_id, default_yes)

        self.params = parse_experiment_config(self.experiment_id)

        set_visible_gpu(self.params['gpu_id'])

        initialize_logger(logdir=None, loglevel=self.params['loglevel'], only_console=True)
        logging.debug('Reading the parameters was successful.')

    def run(self):
        create_crops_based_on_dataset(self.params['crop_id'], self.params['dataset_id'])


if __name__ == '__main__':
    ppr = CropRunner()
    try:
        ppr.run()
    except Exception as e:
        logging.error(str(e))
        raise Exception(e.args)
