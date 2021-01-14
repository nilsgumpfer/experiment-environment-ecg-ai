import logging
import os
import sys

from utils.file.file import cleanup_directory, logdir_exists


def get_experiment_logdir(experiment_id):
    return '../../logs/experiments/{}'.format(experiment_id)


def create_log_directory(experiment_id, category='experiments', default_yes=False):
    if not os.path.exists(os.path.abspath('../../logs/{}'.format(category))):
        os.makedirs(os.path.abspath('../../logs/{}'.format(category)))

    logdir, exists = logdir_exists(experiment_id, category)

    if exists:
        if default_yes in (False, None):
            value = input(
                'Run "{}" ({}) has already been performed. Override? (Y)es (N)o \n>>>'.format(experiment_id, category))
            if value == 'Y' or value == 'y':
                cleanup_directory(logdir)
            else:
                raise ValueError('Run has already been performed. Aborting.')
        else:
            cleanup_directory(logdir)

    if not os.path.exists(os.path.abspath(logdir)):
        os.makedirs(os.path.abspath(logdir))

    return logdir


def initialize_logger(logdir, loglevel='INFO', append=False, only_console=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'

    formatter = logging.Formatter('%(asctime)-15s %(levelname)s %(message)s')

    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(formatter)

    if only_console is False:
        filehandler = logging.FileHandler(logdir + '/logfile.log', mode)
        filehandler.setFormatter(formatter)

    log = logging.getLogger()

    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)

    if only_console is False:
        log.addHandler(filehandler)

    log.addHandler(consolehandler)

    log.setLevel(loglevel)


def print_progress_bar(iteration, total, decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration + 1) / float(total)))
    filled = int(length * (iteration + 1) / total)
    bar = fill * filled + '-' * (length - filled)

    sys.stdout.write('\r{} {}%'.format(bar, percent))
    sys.stdout.flush()

    if iteration + 1 == total:
        sys.stdout.write('\n')
