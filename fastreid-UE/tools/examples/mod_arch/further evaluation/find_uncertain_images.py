# encoding: utf-8
"""
@author:  Andreas Gebhardt
@contact: AGebhardt1999@gmail.com
"""

import itertools
import sys
import os
from nicr_cluster_utils.datasets import load_dataset
from nicr_cluster_utils.utils.job_information import get_job_id, get_array_id
import numpy as np

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import UncertaintyTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
import fastreid.data.build as fdb
from contextlib import contextmanager
import time
import datetime
from fastreid.utils import comm
import logging
from fastreid.utils.logger import log_every_n_seconds

import torch

###############################################################
# -------------------------------------------------------------

def change_out_dir(cfg):
    pass

# -------------------------------------------------------------
###############################################################


def inference_on_dataset_get_extreme_images(model, data_loader, evaluator, flip_test=False, num_extreme=10, extreme_metric="norm"):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """

    def process(self, inputs, outputs):
        # outputs is now a dict
        mu = outputs["mean"]
        variance = outputs["variance"]

        # maybe we will extract more values here if we need e.g. model uncertainty for evaluation
        # then we will also nee to thread those through to the distance calculation in evaluate(), like done with sig and mu

        prediction = {
            'mu': mu.to(self._cpu_device, torch.float32),
            'variance': variance.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'img_paths': inputs['img_paths']

        }
        # TODO add model uncertainty logging
        if "variance_of_mean" in outputs: # TODO: do I need enable_3d here? can i replace it all with "x in dict" checks?
            prediction["variance_of_mean"] = outputs["variance_of_mean"].to(self._cpu_device, torch.float32)
        self._predictions.append(prediction)

    evaluator.process = process


    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                if model.__class__.__name__ == "Uncertainty":
                    raise NotImplementedError("fastreid-UE.fastreid.evaluation.evaluator.inference_on_dataset: flip_test not implemented for handling of uncertainty values.")
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(evaluator, inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    #################################
    #### TODO: use evaluator here. predictions are saved. use them to find high uncertainty images and save them or their paths. ########
    #################################

    def get_predictions(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        mus = []
        variances = []
        variances_of_means = []
        pids = []
        camids = []
        img_paths = []
        for prediction in predictions:
            mus.append(prediction['mu'])
            variances.append(prediction['variance'])
            if "variance_of_mean" in prediction:
                variances_of_means.append(prediction["variance_of_mean"])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            img_paths += prediction['img_paths']

        mus = torch.cat(mus, dim=0)
        variances = torch.cat(variances, dim=0)
        if len(variances_of_means) > 0:
            variances_of_means = torch.cat(variances_of_means, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()

        return {
            'mus': mus,
            'variances': variances,
            'variances_of_means': variances_of_means if len(variances_of_means) > 0 else None,
            'pids': pids,
            'camids': camids,
            'img_paths': img_paths
            }
    

    evaluator.get_predictions = get_predictions
    predictions = evaluator.get_predictions(evaluator)

    mus = predictions['mus']
    variances = predictions['variances']
    variances_of_means = predictions['variances_of_means']
    pids = predictions['pids']
    camids = predictions['camids']
    img_paths = predictions['img_paths']

    variance_norms = torch.linalg.norm(variances, dim=1).squeeze()

    variance_avgs = torch.mean(variances, dim=1).squeeze()

    ret = {}

    for variance_metrics, name in zip([variance_norms, variance_avgs], ["norm", "avg"]):

        # Convert variance_norms to a numpy array for sorting
        variance_norms_np = variance_metrics.cpu().numpy()

        # Sort the indices based on the L2 norms
        sorted_indices = np.argsort(variance_norms_np)

        # Sort the image paths based on the sorted indices
        sorted_img_paths = [img_paths[i] for i in sorted_indices]

        most_uncertain_images = sorted_img_paths[-num_extreme:]
        least_uncertain_images = sorted_img_paths[:num_extreme]

        ret[name] = (most_uncertain_images[::-1], least_uncertain_images) # most and least are at respective index 0

    return ret





@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)







def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    change_out_dir(cfg)

    cfg.freeze()

    dataset_path = load_dataset(cfg.DATASETS.NAMES[0]).as_posix() # could make this more general but fine for now
    fdb._root = dataset_path # will throw warning but work, with Market at least

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = UncertaintyTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        data_loader, evaluator = UncertaintyTrainer.build_evaluator(cfg, "Market1501", last_eval=False)

        ret = inference_on_dataset_get_extreme_images(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED, num_extreme=3)


        for name in ["norm", "avg"]:

            most_uncertain_images, least_uncertain_images = ret[name]

            print(f"Most uncertain images: {most_uncertain_images}\n\n\nleast uncertain images: {least_uncertain_images}\n\n\n first entry is most extreme.")


            import os, shutil
            #this_dir = os.path.dirname(os.path.abspath(__file__))

            this_dir = "/home/ange8547/UE/misc/uncertain_images"

            # thisdir/PFE/avg/most

            os.makedirs(f"{this_dir}/{cfg.MODEL.VAR_HEAD.NAME[:3]}/{name}/most_uncertain", exist_ok=True)
            os.makedirs(f"{this_dir}/{cfg.MODEL.VAR_HEAD.NAME[:3]}/{name}/least_uncertain", exist_ok=True)

            for index, path in enumerate(most_uncertain_images):
                split_path = path.split("/")
                new_path = f"{this_dir}/{cfg.MODEL.VAR_HEAD.NAME[:3]}/{name}/most_uncertain/{index}__{split_path[-1]}"
                shutil.copy(path, new_path)

            for index, path in enumerate(least_uncertain_images):
                split_path = path.split("/")
                new_path = f"{this_dir}/{cfg.MODEL.VAR_HEAD.NAME[:3]}/{name}/least_uncertain/{index}__{split_path[-1]}"
                shutil.copy(path, new_path)
                

    else:
        raise Exception("always eval mode")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    args.eval_only = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


# need to give config file and MODEL.WEIGHTS pointing to a fully trained model