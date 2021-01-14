from evaluation.abstract_evaluator import AbstractEvaluator
from utils.experiments.evaluation import save_results_for_experiment, distribute_results_for_experiment, \
    save_placeholder_files_to_logdir, cleanup_weights_for_experiment, perform_test_for_models


class BasicEvaluator(AbstractEvaluator):

    def __init__(self, params, experiment_id):
        super().__init__(params, experiment_id)

    def perform_evaluation(self):
        save_results_for_experiment(experiment_id=self.experiment_id,
                                    class_names=self.params['class_names'],
                                    calculation_methods=self.params['calculation_methods'],
                                    metrics=self.params['metrics'],
                                    target_metric=self.params['target_metric'],
                                    metric_thresholds={'sensitivity': self.params['sensitivity_threshold'], 'specificity': self.params['specificity_threshold']},
                                    save_raw_results=self.params['save_raw_results'])

        distribute_results_for_experiment(experiment_id=self.experiment_id,
                                          class_names=self.params['class_names'],
                                          calculation_methods=self.params['calculation_methods'],
                                          recipients=self.params['recipients_emails'])

        save_placeholder_files_to_logdir(experiment_id=self.experiment_id, placeholders=['conclusion', 'interpretation'])

        cleanup_weights_for_experiment(experiment_id=self.experiment_id,
                                       class_names=self.params['class_names'],
                                       calculation_methods=self.params['calculation_methods'],
                                       epochs=self.params['number_epochs'])

        perform_test_for_models(experiment_id=self.experiment_id,
                                params=self.params)
