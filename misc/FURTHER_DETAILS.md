
# Further Details

## General Information about FastReID

In fastreid, a model consists of a meta-architecture, i.e. a high-level description of the model, and certain components that are "plugged into" it. In the most basic example, there is just a backbone and an embedding head. These are defined in `fastreid-UE/fastreid/modeling`.

For building a model using these components, a config file is used. These are stored in `fastreid-UE/configs`. 

For starting a training, a `tools` script is used.

There are trainer and evaluator componentes (among others) that can be customized. See below for details on fastreid filestructure to gain an overview over which components there are.

Wandb integration is implemented. Use `wandb.init` in the tool script and logging will be handled automatically.

The default dataset loading mechanism has been overwritten in most of our experiments to enable more flexible use with load sharing facilitators like IBM LSF and SLURM. See tool scripts for examples and the original fastreid for alternatives.

## Working with the Uncertainty Meta-Architecture

The Uncertainty meta-architecture is implemented in `fastreid-UE/fastreid/modeling/meta_arch/uncertainty.py`. It provides places for five components:
- Backbone: feature extraction
- Neck: pre-processing (Bayesian module)
- Mean Head: for computing mean features (probabilistic embedding)
- Var Head: for computing variance features (probabilistic embedding)
- Crown: post-processing (sampling, FV weighting, ...)

It has to be used in concert with the uncertainty trainer and uncertainty evaluator. For examples, see the provided tool scripts.

The output formats are important, as they need to match processing within `uncertainty.py` as well as the evaluator. 

## File Structure

### fastreid-UE
This is the main folder for the adapted FastReID framework.

#### configs
This folder contains the config files. The ones relevant for the uncertainty estimation architectures can be found in `uncertainty`.

#### datasets
This folder is supposed to contain the datasets. A more flexible integration has been used for Market-1501 in the `tools` scripts.

#### fastreid

This folder contains the files of FastReID framework. Note that new components must be registered, not just in the corresponding registry but also in the `__init__` files for importing.
- `config`: definition and parsing of config options. See also: [config documentation](fastreid-UE/fastreid/config/config_option_documentation.md)
- `data`: files relating to data reading, augmentation, and preprocessing.
- `engine`: contains Trainers (the "engine"). For this work a new trainer `uncertaintyTrainer` was developed.
- `evaluation`: files relating to evaluation. For this work a new evaluator `uncertainty_reid_evaluation` was written.
- `layers`: definition of layers, e.g. softmax, batch_norm, pooling, SE, etc. For this work the UAL Bayesian layers were integrated here.
- `modeling`: higher-level model components. 
  - `backbones`: feature extractors, e.g. bag of tricks
  - `crowns`: for the new meta-architecture, post-processing-components
  - `heads`: both normal embedding heads and the mean/var heads for the new meta-architecture
  - `losses`: definition of loss functions
  - `meta_arch`: meta architectures defines a high-level structure for the model where the other component types are plugged in. `uncertainty` was developed in this work
  - `necks`: pre-processing components for the new meta-architecture, e.g. the Bayesian Module from UAL

#### solver
This folder contains optimizers, e.g. Adam.

#### utils
General utilities, e.g. logging.

#### tools
Tool scripts are used to kick off the trainings. 

### misc
This folder contains miscellaneous scripts.

### trained_model
This folder contains a checkpoint and raw model outputs file of a trained UAL/UBER model.

### conda_env
This folder contains the required conda environment.

### distractor_sets.json
A dict mapping set IDs to file names of distractors from Market-1501's `bounding_box_test` set for the degrees of out-of-distribution-ness described in the paper.