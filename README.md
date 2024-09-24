# Improving Re-Identification by Estimating and Utilizing Diverse Uncertainty Types for Embeddings

This is the repository containing the code and best model for the paper <!--[-->Improving Re-Identification by Estimating and Utilizing Diverse Uncertainty Types for Embeddings<!--](TODO: add open access link)--> published in Algorithms. 

Below, you will find instructions for evaluating our model and reproducing our results.

If you use our work, please cite it as follows:<!-- TODO: add other bigliographic data -->
```bibtex
@article{eisenbach2024improving,
  title={Improving Re-Identification by Estimating and Utilizing Diverse Uncertainty Types for Embeddings},
  author={Eisenbach, Markus and Gebhardt, Andreas and Aganian, Dustin and Gross, Horst-Michael},
  journal={Algorithms},
  year={2024},
  publisher={MDPI}
}
```

## Reproduction of Results
The pipeline to reproduce our results consists of three steps:
1. Train the model.
2. Generate a file containing the model outputs for each query and gallery image.
3. Compute performance metrics based on that file.

Depending on how deep you want to get into the reproduction process, you can enter at any step with the material we provided.

### Setup
First, you will have to set up the conda environment.
We were using conda version `23.7.4`. 
Run the following commands:

```bash
conda env create --file conda_env/UE.yml
conda activate UE
```
Please note that we also provide other conda environments that are described [here](conda_env/README.md).

Next, make sure you have [downloaded the Market-1501 dataset](https://www.kaggle.com/datasets/pengcw1/market-1501/data) and have the path ready.

### Training

Should you wish to train a model from scratch, follow these steps:
1. Update the paths in `fastreid-UE/tools/publication/training.py` as needed.
2. Run the following commands:

```bash
cd fastreid-UE
python tools/publication/training.py
```

By default, the model just trains and does no evaluations until it is finished. Should you wish to change this behaviour, edit `volatile_config()` as needed.


### Generation of Model Outputs

If you trained the model from scratch, it has already generated the model outputs file and you can move on with this step.

In case you preferred to skip to train the model yourself, we provide the [checkpoint file for a trained model that can be downloaded](https://drive.google.com/uc?export=download&id=1z8SWm0O6ciwI02RAV5rPWhG5qp12_jM2) and stored as `trained_model/model_best.pth`.
This is the best of our trained models when using the embedding refinement method that produces the best result on average ($D^{\mu,\Sigma^{(M)}}, c = \lambda_{opt} \cdot \frac{1}{\left\|\Sigma^{(V)}\right\|_{1}}$).

If you want to generate the model outputs file based on an existing checkpoint file (either from your training or downloaded), follow these steps
1. Update the paths in `fastreid-UE/tools/publication/training.py` as needed.
2. Run the following commands:

```bash
cd fastreid-UE
python tools/publication/training.py --eval-only
```


### Evaluation based on Model Outputs

With the model outputs file, you can start the evaluation.
In case you preferred to skip the previous steps, we also provide a [model outputs file that can be downloaded](https://drive.google.com/uc?export=download&id=1ezEztkDU8V1NJArvkiAqUjyp_up01G7D) and strored as `trained_model/raw_model_outputs.json`.
If you are interested only in using a trained model, please follow step 2.

In order to evaluate the model, follow these steps:
1. Update the paths in `fastreid-UE/tools/publication/evaluation.py` as needed.
2. Run the following commands:

```bash
cd fastreid-UE
python tools/publication/evaluation.py
```

The results are shown in the console. For the provided model, we get the following outputs:

| Variant                            | mAP [%]  | rank-1 [%]  |
|------------------------------------|----------|-------------|
| UAL                                | 86.9965  | 94.5962     |
| UBER: $\quad D^{\mu,\Sigma^{(M)}}, c = 0.1225$              | 87.4999  | 94.6259     |
| UBER: $\quad D^{\mu,\sqrt{\Sigma^{(M)}}}, c = 0.2435$        | 87.4747  | 94.5962     |
| UBER: $\quad D^{\mu,\Sigma^{(M)}}, c = 1024 \cdot \frac{1}{\lVert\ln\Sigma^{(M)}\rVert_{1}}$      | 87.5022  | 94.6853     |
| UBER: $\quad D^{\mu,\Sigma^{(M)}}, c = 0.0469 \cdot \frac{1}{\lVert \Sigma^{(V)}\rVert_{1}}$      | 87.5006  | 94.5071     |

We have selected a few variants of our approach that are evaluated by default.
Should you wish to examine other variants, edit the code as needed.


## Distractor Sets

We provide the sets of distractor images we have labeled for our experiments.
The file `distractor_sets.json` contains a JSON dict that maps the set ID also used in the paper (D1 - D4) to the filenames of the corresponding images in the `bounding_box_test` partition of Market-1501.

The sets represent increasing degrees of out-of-distribution-ness compared to the training data.

NOTE: The annotations are bound to contain labeling errors. The distractor set also contains ambiguous images.

## Further Information

For further information useful for development based on this repo, see [Further Details](misc/FURTHER_DETAILS.md).

## Attributions
This repository is based on [FastReID](https://github.com/JDAI-CV/fast-reid/tree/master/fastreid).

Adaptations of [UAL](https://github.com/dcp15/UAL/tree/master), [DistributionNet](https://github.com/TianyuanYu/DistributionNet), and [PFE](https://github.com/seasonSH/Probabilistic-Face-Embeddings) are contained here. 

## License
This software is licensed under the [Apache 2.0](LICENSE) license.
