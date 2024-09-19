#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long' ('short' is for testing).
#SBATCH -p long

# We will need HH:MM:SS of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#SBATCH -t 01:45:00

# We want to be notified by email when the jobs starts / ends
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=andreas.gebhardt@tu-ilmenau.de

# specify output directory where console outputs are written to
#SBATCH --output=/results_nas/ange8547/mod_arch/DNet/experiments/slurm_1/slurm-%x-%a-%j.out

# specify job name that is seen in slurm info
#SBATCH --job-name=DNet_BoT_experiments

# make this into an array of jobs
#SBATCH --array=1-14

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
from fastreid.engine import UncertaintyTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
import fastreid.data.build as fdb

import wandb
import torch

###############################################################
# -------------------------------------------------------------

WANDB_PROJECT = "mod_arch-DNet_experiments"

EXPERIMENT_NAMES = [
    "None",
    "TripletL",
    "CenterL",
    "BNNeck",
    "warmup",
    "REA",
    "PaddingA",
    "all",
    "not_TripletL",
    "not_CenterL",
    "not_BNNeck",
    "not_warmup",
    "not_REA",
    "not_PaddingA"
]

def setup_triplet(cfg): cfg.MODEL.LOSSES.NAME += ("TripletLoss",)
def setup_center(cfg): cfg.MODEL.LOSSES.NAME += ("CenterLoss",) 
def setup_bnneck(cfg): 
    cfg.MODEL.CROWN.WITH_BN = True
    for key in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        cfg.MODEL.CHECKPOINT_REMAP.append([f"heads.bottleneck.0.{key}", f"crown.norm.{key}"])
def setup_warmup(cfg): cfg.SOLVER.WARMUP_ITERS = 800
def setup_rea(cfg): cfg.INPUT.REA.ENABLED = True
def setup_padding(cfg): cfg.INPUT.PADDING.ENABLED = True

EXPERIMENT_SETUP_FUNCTIONS = [
    setup_triplet,
    setup_center,
    setup_bnneck,
    setup_warmup,
    setup_rea,
    setup_padding
]

def multi_setup(cfg, indices):
    for index in indices:
        EXPERIMENT_SETUP_FUNCTIONS[index](cfg)

# want to turn on some of the config options in each experiment. Which ones are regulated here using the index for the functions list
EXPERIMENT_SETUP_INDEX_MAP = [
    [],
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [0,1,2,3,4,5],
    [1,2,3,4,5],
    [0,2,3,4,5],
    [0,1,3,4,5],
    [0,1,2,4,5],
    [0,1,2,3,5],
    [0,1,2,3,4],
]


def volatile_config(cfg):

    #cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler" # naive fine tune was the best

    cfg.SEED = 11864212

    slurm_array_id = get_array_id()
    if slurm_array_id != None:

        OUTPUT_ADDENUM = os.path.join("experiments", EXPERIMENT_NAMES[int(slurm_array_id)-1])

        PRETRAIN_ADDENUM = os.path.join("naive", "8", "model_best.pth") # naive BB was the best, choose arbitrary example (good performance)

        multi_setup(cfg, EXPERIMENT_SETUP_INDEX_MAP[int(slurm_array_id)-1]) # set up the experiment

        
        # set identifying config options
        cfg.run_type = "dnet_" + EXPERIMENT_NAMES[int(slurm_array_id)-1]

        print("Slurm Array ID found!")
        print("new output dir:", cfg.OUTPUT_DIR, "+", OUTPUT_ADDENUM)
        print("new pretrain dir:", cfg.MODEL.WEIGHTS, "+", PRETRAIN_ADDENUM)

        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, OUTPUT_ADDENUM)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.MODEL.WEIGHTS, PRETRAIN_ADDENUM)
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
        model = UncertaintyTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = UncertaintyTrainer.test(cfg, model)
        return res

    trainer = UncertaintyTrainer(cfg)

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
