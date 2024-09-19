"""
@author:  Andreas Gebhardt
@contact: AGebhardt1999@gmail.com
"""

import os
import sys
sys.path.append('.')

import fastreid.data.build as fdb
from fastreid.config import get_cfg
from fastreid.engine import (UncertaintyTrainer, default_argument_parser,
                             default_setup, launch)
from fastreid.utils.checkpoint import Checkpointer

# ********************************************************************************
# Update as needed
# ********************************************************************************
MARKET_PATH = "path/to/Market-1501"
OUTPUT_PATH = "../custom_model"
CHECKPOINT_PATH = "../trained_model/model_best.pth"
# CHECKPOINT_PATH = "../custom_model/model_best.pth"

CONFIG_PATH = "configs/uncertainty/UAL.yml" # specify config path in code. You can overwrite by using --config-file in console.
RUN_TYPE = "UBER"
# ********************************************************************************


def volatile_config(cfg):
    """
    In order to facilitate rapid experimentation, the configuration is split into
    persistent and volatile configs.
    Persistent configs are stored in `.yml`-files in the `configs` folder and parsed from there.
    Volatile configs are the differences from that which define your experiment. E.g. you
    might want to try out a different learning rate, so you use the same persistent config
    as before but change the learning rate here.
    """

    cfg.OUTPUT_DIR = OUTPUT_PATH
    cfg.TEST.EXTRA_END_EVAL = False
    cfg.SOLVER.CHECKPOINT_PERIOD = 120 
    cfg.TEST.EVAL_PERIOD = 120 
    # cfg.TEST.EVAL_ENABLED_IN = "0,10,20,30,40,45-55,60,70,75,80,84,88,92,94,96,98,100-120"

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    volatile_config(cfg)

    cfg.run_type = RUN_TYPE # set identifying config option

    cfg.freeze()

    fdb._root = MARKET_PATH # will throw warning but work, with Market at least
    
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = UncertaintyTrainer.build_model(cfg)

        Checkpointer(model, checkpoint_remap=cfg.MODEL.CHECKPOINT_REMAP).load(CHECKPOINT_PATH)  # load trained model

        trainer = UncertaintyTrainer(cfg)

        res = trainer.test(cfg, model)
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
