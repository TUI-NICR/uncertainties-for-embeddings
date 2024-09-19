# Config Options

Config options in fastreid must be defined in `defaults.py` before they can be used.

This file serves as an overview and explanation of the possible config options. **It is as yet very incomplete.**

NOTE: Mostly referencing things relating to the Uncertainty meta-architecture. This is not part of the original fastreid.

## MODEL

### FREEZE_WEIGHTS
List of strings. For use with Uncertainty meta-arch. Each string is a path to a module to freeze the weights for. Freezing weights means locking the module in eval mode forever and setting requires_grad to False for all its parameters. "Path" means how you would reference that module in code, based off the meta arch. E.g.: uncertainty.[backbone.layer4] (path marked in square brackets)

### CHECKPOINT_REMAP
List of lists of lenth 3, e.g. `[[heads.weight, crown.logit_layer.weight, 2]]` (see DNet.yml as example)
The inner lists have the format [oldName, newName, n] where n is the number of unsqueeze(-1) to apply (squeeze if negative, squeeze() if 0). If you have a checkpoint you want to load into a slightly modified architecture, this is your friend.

### HEADS
#### PARALLEL_*
The config options
- MODEL.HEADS.PARALLEL_DIM_REPEAT
- MODEL.HEADS.PARALLEL_DIM_REPEAT_EVAL

control how many times the weight sampling in the Bayesian module and subsequent processing is repeated during train and eval respectively. It is NOT in parralel, but rather sequential.

The config options 
- MODEL.HEADS.PARALLEL_3D_CONV
- MODEL.HEADS.PARALLEL_BN_FULL_3D
- MODEL.HEADS.PARALLEL_DIM_LEN
- MODEL.HEADS.PARALLEL_DIM_LEN_EVAL

were part of a failed experiment to parallelize the piepeline for UAL with multiple weight samples in the Bayesian module. If you are interested, look at the code. Caution advised, it produced untraceable errors on our hardware. See `layers/UAL.py`.

### CROWN, NECK, MEAN_HEAD, VAR_HEAD
These control which components are used in the architecture and their options. Just look at how they are used, it's pretty straightforward. 

### LOSSES
There are various losses that have been added. Their config options are stored here. See `uncertainty.py` for details.

### DYNAMIC_CONFIG
Dynamically change config of model during training:
[[epoch, name, value], [2, "loss_kwargs.mls.scale", 1337], ...]

The "paths" are similar to the way they are used in FREEZE_WEIGHTS.

WARNING: No guarantees that this works as it was abandoned.

## TEST

### EVAL_ENABLED_IN
With this option, we can specify epochs during which evaluation (and checkpointing) is enabled. The default value is `None` meaning all epochs.

The option is a string like such: `"0,10,20-30,100-120"`

If the current epoch number is one of the given ones or in one of the given ranges, evaluation is possible. However, the other conditions also still apply. So if at the same time `EVAL_PERIOD` is set to `2`, evaluation would still only happen every other epoch in the given ranges. 

The purpose of this option is to disable evaluation during the early training stages when we are really only interested in the detailed behaviour of the performance during the later training stages.

An easy way to use this option is to set `CHECKPOINT_PERIOD` and `EVAL_PERIOD` to `1` and then manually define all the desired epochs where evaluation should occur with this option.

### EXTRA_END_EVAL
Do you want to do an extra evaluation at the end of the training with more distance functions? This is how. Set this to true and specify `EXTRA_END_EVAL_METRIC: [cosine, sigma_cosine, sqrt_sigma_cosine, euclidean, sigma_euclidean, sqrt_sigma_euclidean, kl_divergence, js_divergence, bhat_distance, wasserstein]` or similar in the config (e.g. `configs/uncertainty/UAL.yml`).

For development: see `uncertainty_reid_evaluation.py`

### UNCERTAINTY
To get a sense for what the uncertainty values are doing, you can log some stuff with this. `configs/uncertainty/UAL.yml` uses everything there is:
```
  UNCERTAINTY:
    MODEL:
      SETS: ["Q", "G", "Q+G", "D1", "D2", "D3", "D4"]
      AGGREGATIONS: ["min", "max", "avg"]
      FUNCTIONS: [1, 2, "max", "min", "avg", "entropy"]
    DATA:
      SETS: ["Q", "G", "Q+G", "D1", "D2", "D3", "D4"]
      AGGREGATIONS: ["min", "max", "avg"]
      FUNCTIONS: [1, 2, "max", "min", "avg", "entropy"]
    DIST:
      SETS: ["Q", "G", "Q+G", "D1", "D2", "D3", "D4"]
      AGGREGATIONS: ["min", "max", "avg"]
      FUNCTIONS: [1, 2, "max", "min", "avg", "entropy"]
```

For the respective uncertainty type, take all uncertainty vectors of the specified set, apply the function to each of them (1,2 = L1/L2-Norm, you can use any int/float that `torch.linalg.vector_norm` accepts), and aggregate the resulting scores over the set using the specified aggegation. All combinations are used.

Also for each of these, an entry is made in the most uncertain images JSON-output. A JSON object is exported containing NUM_UNCERTAIN_IMAGES image names for each of the combinations. This is exported:

```
uncertain_images[uncertainty_type][set_id][function_label] = {
    "most_uncertain": most_uncertain_images,
    "most_uncertain_scores": most_uncertain_scores,
    "medium_uncertain": medium_uncertain_images,
    "medium_uncertain_scores": medium_uncertain_scores,
    "least_uncertain": least_uncertain_images,
    "least_uncertain_scores": least_uncertain_scores
}
```
NOTE: most uncertain is last entry in list, least uncertain is first entry in list. Basically as if plucked from the long list of all scores.

See also `uncertainty_reid_evaluation.py`

## SEED
With this option, we can control what seed is used to generate randomness. 

Usually, the `default_setup` function from `fastreid/engine/defaults.py` is called in the tool script. Among other things, it calls the `seed_all_rng` function from `fastreid/utils/env.py` which seeds the rng for `numpy`, `torch`, and `random`.

If this option is `None`, a random seed is generated.

If this option is an `int`, it is used as the seed.

For distributed computation, it is also possible to set this option to a list `[(rank1, seed1), (rank2, seed2), ...]` where `rank` is a unique identifier for each involved process. The ranks are consecutive integers from 0 to `world_size` (total number of processes). If you want to see this in detail, do a test run where the seeds are randomly generated and then look at the resulting `SEEDS_USED` config option (see below).

## SEEDS_USED
**This option is not to be set manually!**

For logging purposes, this option is used to store every seed that every involved process uses (in case of distributed computation, this might be more than one). 

The format is `[(rank1, seed1), (rank2, seed2), ...]` where `rank` is a unique identifier for each involved process. The ranks are consecutive integers from 0 to `world_size` (total number of processes).

## KEEP_ONLY_BEST_CHECKPOINT

Normally, FastReID would save a checkpoint every `SOLVER.CHECKPOINT_PERIOD` epochs and additionally keep a `model_best.pth` with the checkpoint that produced the best results and `model_final.pth` for the final checkpoint.

If this option is set to `True`, we still try to create a checkpoint every `SOLVER.CHECKPOINT_PERIOD` epochs but we only ever have one checkpoint: `model_best.pth` which is only updated if the current performance is better than the previous best one. 

This is useful to save disk space and avoid cleanup work. Essentially, the behaviour is exactly the same as before, except that saving of (e.g.) `model_0001.pth` and `model_final.pth` is omitted.

**NOTE:** In order for FastReID to know that the current model is better than the previous best checkpoint, it needs to have done an evaluation recently. The metric used to determine whether the current model is better than the previous best checkpoint is called `metric` and taken from the latest evaluation. So take care to set `TEST.EVAL_PERIOD` accordingly (you probably want the two periods to be equal).

This behaviour is implemented in `PeriodicCheckpointer.step()` in `fastreid/utils/checkpoint.py`.