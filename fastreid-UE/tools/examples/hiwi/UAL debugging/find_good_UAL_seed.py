#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long' ('short' is for testing).
#SBATCH -p long

# We will need HH:MM:SS of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#SBATCH -t 04:15:00

# We want to be notified by email when the jobs starts / ends
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=andreas.gebhardt@tu-ilmenau.de

# specify output directory where console outputs are written to
#SBATCH --output=/results_nas/ange8547/hiwi/UAL/Dropout/%a/slurm-%j.out

# specify job name that is seen in slurm info
#SBATCH --job-name=UAL_baseline_30_2D_

# make this into an array of jobs
#SBATCH --array=5-9

# encoding: utf-8
"""
@author:  Andreas Gebhardt
@contact: AGebhardt1999@gmail.com
"""

import os
import sys
import torch
import wandb

from nicr_cluster_utils.datasets import load_dataset
from nicr_cluster_utils.utils.job_information import get_array_id, get_job_id
from nicr_cluster_utils.utils.wandb_integration import is_wandb_available

sys.path.append('.')
import fastreid.data.build as fdb
from fastreid.config import get_cfg
from fastreid.engine import (UncertaintyTrainer, default_argument_parser,
                             default_setup, launch)
from fastreid.utils.checkpoint import Checkpointer

is_test_run = False

###############################################################
# -------------------------------------------------------------
RUN_TYPE = "test"
WANDB_PROJECT = "UAL_debugging"

CONFIG_PATH = "configs/uncertainty/UAL.yml" # specify config path in code. You can overwrite by using --config-file in console.


def volatile_config(cfg):
    """
    In order to facilitate rapid experimentation, the configuration is split into
    persistent and volatile configs. 
    Persistent configs are stored in .yml-files in the configs folder and parsed from there. 
    Volatile configs are the differences from that which define your experiment. E.g. you
    might want to try out a different learning rate, so you use the same persistent config
    as before but change the learning rate here.
    """

    cfg.SOLVER.CHECKPOINT_PERIOD = 1
    cfg.TEST.EVAL_PERIOD = 1
    cfg.TEST.EXTRA_END_EVAL = True
    cfg.OUTPUT_DIR = "/usr/scratch4/angel8547/results/testing"


    # load orig checkpoint
    """
    cfg.MODEL.WEIGHTS = "/results_nas/ange8547/hiwi/UAL/orig/Dropout_orig/0/model_best.pth"
    cfg.MODEL.CHECKPOINT_REMAP = [["heads.weight", "crown.logit_layer.weight", 2],
                                  ["heads.bottleneck.0.*", "crown.bottleneck.BN.*"],
                                  ["heads.bayes_head.*", "neck.*"],
                                  ["heads.mean_head.*", "mean_head.*"],
                                  ["heads.var_head.*", "var_head.*"]]
    for i in range(3):
        cfg.MODEL.CHECKPOINT_REMAP.append([f"neck.layer.BN_{i}.*", f"neck.layer.BN_{i}.BN.*"])
        cfg.MODEL.CHECKPOINT_REMAP.append([f"mean_head.layer.BN_{i}.*", f"mean_head.layer.BN_{i}.BN.*"])
        if i < 2:
            cfg.MODEL.CHECKPOINT_REMAP.append([f"var_head.layer.BN_{i}.*", f"var_head.layer.BN_{i}.BN.*"])

    for name in ["var_head.layer.Conv_0.weight", "mean_head.layer.Conv_2.weight", "crown.logit_layer.weight", "var_head.layer.Conv_1.weight", "mean_head.layer.Conv_1.weight", "mean_head.layer.Conv_0.weight"]:
        cfg.MODEL.CHECKPOINT_REMAP.append([name, name, 1])
    """

 # startcommand: bsub -gpu "num=1" -J "name[1-10]" -outdir /usr/scratch4/angel8547/results/testing/Dropout/%I -o /usr/scratch4/angel8547/results/testing/Dropout/%I/out.%J -e /usr/scratch4/angel8547/results/testing/Dropout/%I/err.%J python tools/hiwi/UAL\ debugging/find_good_UAL_seed.py

    

    cfg.TEST.EVAL_ENABLED_IN = "0,10,20,30,40,45-55,60,70,75,80,84,88,92,94,96,98,100-120"

    slurm_array_id = get_array_id()

    if "[" in os.environ["LSB_BATCH_JID"]:
        # LSF array
        slurm_array_id = str(os.environ["LSB_JOBINDEX"]) # 0 if not used

    print("ID: ", slurm_array_id)

    if slurm_array_id == None:
        # if we are not in slurm, this is a test
        slurm_array_id = "test"
        global is_test_run
        is_test_run = True
    #else:
    #    slurm_array_id = str(int(slurm_array_id)-1) # id=0 messes with queue viewer website but want folder names 0..9

    if slurm_array_id != None:

        OUTPUT_ADDENUM = os.path.join("Dropout", slurm_array_id)

        
        print("new output dir:", cfg.OUTPUT_DIR, "+", OUTPUT_ADDENUM)

        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, OUTPUT_ADDENUM)

# -------------------------------------------------------------
###############################################################



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    volatile_config(cfg)

    # set identifying config options
    cfg.run_type = RUN_TYPE

    cfg.freeze()

    try:
        dataset_path = load_dataset(cfg.DATASETS.NAMES[0]).as_posix() # could make this more general but fine for now
    except FileNotFoundError:
        print("nicr_cluster_utils.datasets.load_dataset failed. Using custom path instead.")
        dataset_path = f"/usr/scratch4/angel8547/datasets/{cfg.DATASETS.NAMES[0]}"
    fdb._root = dataset_path # will throw warning but work, with Market at least

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # if connection is available, sync directly, otherwise store data for later sync-ing
    try:
        wandb_available = is_wandb_available()
    except Exception:
        # if check fails, better to be prudent
        wandb_available = False
    if wandb_available:
        mode = "online"
        print("wandb online mode")
    else:
        mode = "offline" 
        print("wandb offline mode")

    if not is_test_run:
        # initialize wandb logging
        wandb.init(
            project=WANDB_PROJECT,
            config=cfg,
            mode=mode
        )

        # customize logging behaviour
        # We can specify (combinations of) certain aggregations to be shown in summary
        # Valid aggregations are: "min", "max", "mean", "best", "last", "copy", "none" (e.g. "max,last" is also possible for showing both)
        # For details, see: https://docs.wandb.ai/guides/track/log/log-summary#customize-summary-metrics

        # show both latest and max result of eval.cosine.mAP in summary
        wandb.define_metric("eval.cosine.mAP", summary="max,last")


    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = UncertaintyTrainer.build_model(cfg)

        Checkpointer(model, checkpoint_remap=cfg.MODEL.CHECKPOINT_REMAP).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = UncertaintyTrainer.test(cfg, model)
        return res

    trainer = UncertaintyTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":

    # so we don't have to enter the same command line argument every time
    # but are still able to overwrite it
    if not "--config-file" in sys.argv:
        sys.argv = [sys.argv[0]] + ["--config-file", CONFIG_PATH] + sys.argv[1:]

    # argument parsing
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # begin
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
