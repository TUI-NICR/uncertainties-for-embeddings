#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long' ('short' is for testing).
#SBATCH -p long

# We will need HH:MM:SS of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#SBATCH -t 02:00:00

# We want to be notified by email when the jobs starts / ends
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=andreas.gebhardt@tu-ilmenau.de

# specify output directory where console outputs are written to
#SBATCH --output=/results_nas/ange8547/mod_arch/bagtricks/random/%a/slurm-%j.out

# specify job name that is seen in slurm info
#SBATCH --job-name=pretrain_baseline_bagtricks__random

# make this into an array of five jobs
#SBATCH --array=0-9

# encoding: utf-8
"""
@author:  Andreas Gebhardt
@contact: AGebhardt1999@gmail.com
"""

import sys
import os
from nicr_cluster_utils.datasets import load_dataset
from nicr_cluster_utils.utils.wandb_integration import is_wandb_available
from nicr_cluster_utils.utils.job_information import get_job_id, get_array_id

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
import fastreid.data.build as fdb

import wandb

###############################################################
# -------------------------------------------------------------
RUN_TYPE = "pre-bagtricks-random"
WANDB_PROJECT = "mod_arch"

def configure_volatile(cfg):

    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    slurm_array_id = get_array_id()
    if slurm_array_id != None:

        ADDENUM = os.path.join("random", str(slurm_array_id))

        print("Slurm Array ID found!")
        print("new output dir:", cfg.OUTPUT_DIR, "+", ADDENUM)

        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, ADDENUM)
# -------------------------------------------------------------
###############################################################



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    configure_volatile(cfg)

    # set identifying config options
    cfg.run_type = RUN_TYPE

    cfg.freeze()

    dataset_path = load_dataset(cfg.DATASETS.NAMES[0]).as_posix() # could make this more general but fine for now
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
    #mode = "online"

    # initialize wandb logging
    wandb.init(
        project=WANDB_PROJECT,
        config=cfg,
        mode=mode
    )


    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
