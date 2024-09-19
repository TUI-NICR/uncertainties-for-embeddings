"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

import logging
from collections import OrderedDict
import torch

from .defaults import DefaultTrainer
from fastreid.utils import comm
from fastreid.evaluation import UncertaintyReidEvaluator, inference_on_dataset, print_csv_format_multi

class UncertaintyTrainer(DefaultTrainer):
    """
    Essentially the same as DefaultTrainer except that it uses the 
    UncertaintyReidEvaluator to allow for evaluation which uses the uncertainty values.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

    #@classmethod
    def build_evaluator(self, cfg, dataset_name, output_dir=None, last_eval=False):
        data_loader, num_query = self.__class__.build_test_loader(cfg, dataset_name)
        return data_loader, UncertaintyReidEvaluator(cfg, num_query, output_dir, last_eval=last_eval, best_metric=self._periodic_checkpointer_hook.best_metric)
    
    #@classmethod
    def test(self, cfg, model, last_eval=False):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            try:
                data_loader, evaluator = self.build_evaluator(cfg, dataset_name, last_eval=last_eval)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue

            try:
                if model.heads.__class__.__name__ == 'BayesHead': # UAL reproduction
                    with torch.no_grad():
                        model.heads._reset_weight()
            except Exception as e:
                pass

            results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED) # result of evaluator.evaluate()
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(
                    results, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                results_i['dataset'] = dataset_name
                print_csv_format_multi(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        return results







# Access basic attributes from the underlying trainer
# copied from defaults.py, not sure this is necessary
for _attr in ["model", "data_loader", "optimizer", "grad_scaler"]:
    setattr(UncertaintyTrainer, _attr, property(lambda self, x=_attr: getattr(self._trainer, x, None)))