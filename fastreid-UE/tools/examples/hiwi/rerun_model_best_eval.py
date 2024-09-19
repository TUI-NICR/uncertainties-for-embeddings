#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long' ('short' is for testing).
#SBATCH -p long

# We will need HH:MM:SS of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#SBATCH -t 10:00:00

# We want to be notified by email when the jobs starts / ends
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=andreas.gebhardt@tu-ilmenau.de

# specify output directory where console outputs are written to
#SBATCH --output=/results_nas/ange8547/hiwi/UAL/%a/slurm-%j.out

# specify job name that is seen in slurm info
#SBATCH --job-name=UAL

# make this into an array of jobs
#SBATCH --array=10-29

# encoding: utf-8
"""
@author:  Andreas Gebhardt
@contact: AGebhardt1999@gmail.com
"""

import json
import os
import shutil
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

# got seed resulting in 87.06 mAP: 21001654

# TODO: maybe try again with linear instead of 1x1 conv

###############################################################
# bsub -gpu "num=1" -J "UAL[10-29]" -q "BatchGPU" -outdir /usr/scratch4/angel8547/results/UAL/%I -o /usr/scratch4/angel8547/results/UAL/%I/out.%J -e /usr/scratch4/angel8547/results/UAL/%I/err.%J python tools/hiwi/UAL.py
# -------------------------------------------------------------
RUN_TYPE = "UAL"
WANDB_PROJECT = "UAL_augmentations"

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
    #cfg.OUTPUT_DIR = "/usr/scratch4/angel8547/results/UAL" # change out_dir path when using HPC

    #cfg.TEST.EVAL_ENABLED_IN = "0,10,20,30,40,45-55,60,70,75,80,84,88,92,94,96,98,100-120"


    



    job_array_id = get_array_id() # slurm array

    if "LSB_BATCH_JID" in os.environ and "[" in os.environ["LSB_BATCH_JID"]:
        # LSF array
        job_array_id = str(os.environ["LSB_JOBINDEX"]) # 0 if not used

    if job_array_id == None:
        # if we are not in cluster, this is a test
        job_array_id = "test"
        global is_test_run
        is_test_run = True

    if "%I" in cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("%I", job_array_id)

    if "LSB_BATCH_JID" not in os.environ:
        OUTPUT_ADDENUM = job_array_id
        
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

    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/results_nas/ange8547/hiwi", "/usr/scratch4/angel8547/results")

    #volatile_config(cfg)
    cfg.TEST.UNCERTAINTY.DATA.SETS = ["Q", "G", "Q+G", "D", "D1", "D2", "D3", "D4"]
    cfg.TEST.UNCERTAINTY.MODEL.SETS = ["Q", "G", "Q+G", "D", "D1", "D2", "D3", "D4"]
    cfg.TEST.UNCERTAINTY.DIST.SETS = ["Q", "G", "Q+G", "D", "D1", "D2", "D3", "D4"]

    # set identifying config options
    #cfg.run_type = RUN_TYPE

    cfg.freeze()

    try:
        dataset_path = load_dataset(cfg.DATASETS.NAMES[0]).as_posix() # could make this more general but fine for now
        print("Dataset loaded using nicr_cluster_utils.")
    except FileNotFoundError:
        print("nicr_cluster_utils.datasets.load_dataset failed (file not found). Using custom path instead.")
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

    global is_test_run
    if not is_test_run and False:
        # initialize wandb logging
        wandb.init(
            project="test",
            config=cfg,
            mode=mode
        )

        # customize logging behaviour
        # We can specify (combinations of) certain aggregations to be shown in summary
        # Valid aggregations are: "min", "max", "mean", "best", "last", "copy", "none" (e.g. "max,last" is also possible for showing both)
        # For details, see: https://docs.wandb.ai/guides/track/log/log-summary#customize-summary-metrics

        # show both latest and max result of eval.cosine.mAP in summary
        wandb.define_metric("eval.cosine.mAP", summary="max,last")


    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = UncertaintyTrainer.build_model(cfg)

    Checkpointer(model, checkpoint_remap=[]).load(os.path.join(cfg.OUTPUT_DIR, "model_best.pth"))  # load trained model

    trainer = UncertaintyTrainer(cfg)

    res = trainer.test(cfg, model)
    return res



if __name__ == "__main__":

    #121

    PATHS = [None,
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/12",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/16",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/13",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/10",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/17",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/18",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/14",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/15",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/19",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS_raw/11",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/12",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/16",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/13",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/10",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/17",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/18",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/14",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/15",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/19",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_euc/11",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/27",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/29",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/12",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/22",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/16",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/13",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/10",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/25",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/17",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/23",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/30",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/24",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/33",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/18",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/26",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/34",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/28",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/31",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/14",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/32",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/15",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/19",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/20",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/11",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet/21",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/12",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/16",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/13",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/10",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/17",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/18",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/14",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/15",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/19",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE/11",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/12",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/16",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/13",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/10",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/17",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/18",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/14",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/15",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/19",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri/11",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/27",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/29",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/12",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/22",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/16",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/13",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/10",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/25",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/17",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/23",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/30",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/24",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/33",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/18",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/26",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/34",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/28",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/31",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/14",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/32",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/15",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/19",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/20",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/11",
        "/usr/scratch4/angel8547/results/variations/UAL_cDNet_Tensor/21",
        "/usr/scratch4/angel8547/results/variations/UAL_cPFE+Tri_Bhat/10",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/12",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/16",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/13",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/10",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/17",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/18",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/14",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/15",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/19",
        "/usr/scratch4/angel8547/results/variations/UAL+MLS/11",
        "/usr/scratch4/angel8547/results/UAL/27",
        "/usr/scratch4/angel8547/results/UAL/29",
        "/usr/scratch4/angel8547/results/UAL/12",
        "/usr/scratch4/angel8547/results/UAL/22",
        "/usr/scratch4/angel8547/results/UAL/16",
        "/usr/scratch4/angel8547/results/UAL/13",
        "/usr/scratch4/angel8547/results/UAL/10",
        "/usr/scratch4/angel8547/results/UAL/25",
        "/usr/scratch4/angel8547/results/UAL/17",
        "/usr/scratch4/angel8547/results/UAL/23",
        "/usr/scratch4/angel8547/results/UAL/24",
        "/usr/scratch4/angel8547/results/UAL/18",
        "/usr/scratch4/angel8547/results/UAL/26",
        "/usr/scratch4/angel8547/results/UAL/28",
        "/usr/scratch4/angel8547/results/UAL/14",
        "/usr/scratch4/angel8547/results/UAL/15",
        "/usr/scratch4/angel8547/results/UAL/19",
        "/usr/scratch4/angel8547/results/UAL/20",
        "/usr/scratch4/angel8547/results/UAL/11",
        "/usr/scratch4/angel8547/results/UAL/21"
    ]

    UAL_PATHS = [None,
        "/usr/scratch4/angel8547/results/UAL/27",
        "/usr/scratch4/angel8547/results/UAL/29",
        "/usr/scratch4/angel8547/results/UAL/12",
        "/usr/scratch4/angel8547/results/UAL/22",
        "/usr/scratch4/angel8547/results/UAL/16",
        "/usr/scratch4/angel8547/results/UAL/13",
        "/usr/scratch4/angel8547/results/UAL/10",
        "/usr/scratch4/angel8547/results/UAL/25",
        "/usr/scratch4/angel8547/results/UAL/17",
        "/usr/scratch4/angel8547/results/UAL/23",
        "/usr/scratch4/angel8547/results/UAL/24",
        "/usr/scratch4/angel8547/results/UAL/18",
        "/usr/scratch4/angel8547/results/UAL/26",
        #"/usr/scratch4/angel8547/results/UAL/28",
        "/usr/scratch4/angel8547/results/UAL/14",
        "/usr/scratch4/angel8547/results/UAL/15",
        "/usr/scratch4/angel8547/results/UAL/19",
        "/usr/scratch4/angel8547/results/UAL/20",
        "/usr/scratch4/angel8547/results/UAL/11",
        "/usr/scratch4/angel8547/results/UAL/21"
    ]

    if False:
        for path in PATHS:
            if path == None:
                continue

            full_path = os.path.join(path, "uncertain_images.json")

            with open(full_path, "r") as f:
                json_obj = json.load(f)
                new_name = json_obj["name"].replace("/", "-").replace("+", "-") + ".json"

            shutil.copy(full_path, os.path.join("/usr/scratch4/angel8547/results/download", new_name))
            #print("cp", full_path, os.path.join("/usr/scratch4/angel8547/results/download", new_name))
        exit() 

    #for path in ["/usr/scratch4/angel8547/results/UAL/28"]:
    #for path in ["/usr/scratch4/angel8547/results/UAL/18"]:
    #for path in [PATHS[int(str(os.environ["LSB_JOBINDEX"]))]]: # array id in 1-121
    for path in [UAL_PATHS[int(str(os.environ["LSB_JOBINDEX"]))]]: # array id in 1-20 or 1-19

        # so we don't have to enter the same command line argument every time
        # but are still able to overwrite it
        if not "--config-file" in sys.argv:
            sys.argv = [sys.argv[0]] + ["--config-file", os.path.join(path, "config.yaml")] + sys.argv[1:]

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
