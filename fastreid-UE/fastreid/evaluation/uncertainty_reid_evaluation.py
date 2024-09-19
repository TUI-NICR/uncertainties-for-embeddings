import json
import logging
import itertools
import copy
import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
import sklearn

import torch
import torch.nn.functional as F
import wandb

from fastreid.evaluation.reid_evaluation import ReidEvaluator
from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .query_expansion import aqe
from .rank import evaluate_rank

logger = logging.getLogger(__name__)

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

"""
Based on the existing reid_evaluation.py
"""



# map config option to evaluation function
# support vector norms from torch.linalg.vector_norm, average, and entropy
def get_eval_function(id):
    if type(id) == int or type(id) == float:
        return lambda x: torch.linalg.vector_norm(x, dim=1, ord=id)
    elif type(id) == str:
        if id == "max" or id == "inf":
            return lambda x: torch.linalg.vector_norm(x, dim=1, ord=float("inf"))
        elif id == "min" or id == "-inf":
            return lambda x: torch.linalg.vector_norm(x, dim=1, ord=float("-inf"))
        elif id == "avg":
            return lambda x: torch.mean(x, dim=1)
        elif id == "entropy":
            return lambda x: 0.5*torch.sum(torch.log(x), dim=1) + 1.4189385332 * x.shape[1] # multivariate normal with diagonal covariance
        else:
            raise ValueError(f"Uncertainty evaluation function must be int, float, max (=inf), min (=-inf), avg, or entropy but got {id}.")
    else:
                raise ValueError(f"Uncertainty evaluation function must be int, float, max (=inf), min (=-inf), avg, or entropy but got {id}.")


class UncertaintyReidEvaluator(ReidEvaluator):

    def __init__(self, cfg, num_query, output_dir=None, last_eval=False, best_metric=-1):
        super().__init__(cfg, num_query, output_dir=output_dir)
        self.last_eval = last_eval
        self.enable_3d = cfg.MODEL.HEADS.PARALLEL_3D_CONV # do we still need this?
        self.best_metric = best_metric

    def process(self, inputs, outputs):
        # outputs is now a dict
        mean_vector = outputs["mean_vector"]
        variance_vector = outputs["variance_vector"]

        # maybe we will extract more values here if we need e.g. model uncertainty for evaluation
        # then we will also nee to thread those through to the distance calculation in evaluate(), like done with sig and mu

        prediction = {
            'mean_vector': mean_vector.to(self._cpu_device, torch.float32),
            'variance_vector': variance_vector.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'img_paths': inputs["img_paths"]
        }
        
        if "variance_of_mean_vector" in outputs: 
            prediction["variance_of_mean_vector"] = outputs["variance_of_mean_vector"].to(self._cpu_device, torch.float32)
            prediction["variance_of_variance_vector"] = outputs["variance_of_variance_vector"].to(self._cpu_device, torch.float32)
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        prepared_predictions = self.prepare_predictions(predictions)

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            logger.warn("AQE has not been integrated into the uncertainty framework yet. Be careful when using it with statistical distances.") # TODO: do this
            raise NotImplementedError()
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            # TODO: we probably have to sort the other lists as well
            query_mean_vectors, gallery_mean_vectors = aqe(query_mean_vectors, gallery_mean_vectors, qe_time, qe_k, alpha)

        # allow for singular metric or multi-metric evaluation
        if isinstance(self.cfg.TEST.METRIC, str):
            metrics = [self.cfg.TEST.METRIC]
        else:
            metrics = self.cfg.TEST.METRIC

        # allow for a different evaluation at the end of the training
        if self.last_eval: # the evaluator is instantiated anew every evaluation
            # allow for singular metric or multi-metric evaluation
            if isinstance(self.cfg.TEST.EXTRA_END_EVAL_METRIC, str):
                metrics = [self.cfg.TEST.EXTRA_END_EVAL_METRIC]
            else:
                metrics = self.cfg.TEST.EXTRA_END_EVAL_METRIC
                
        self.evaluate_embedding(prepared_predictions, metrics)

        self.evaluate_uncertainty(prepared_predictions, self.best_metric, self._results["metric"]) # todo pull best metric and new metric in here for if

        return copy.deepcopy(self._results)
    
    def prepare_predictions(self, predictions):
        # the things we care about
        mean_vectors = []
        variance_vectors = []
        variance_of_mean_vectors = []
        variance_of_variance_vectors = []
        pids = []
        camids = []
        img_paths = []

        # extract stuff from the preditions
        for prediction in predictions:
            mean_vectors.append(prediction['mean_vector'])
            variance_vectors.append(prediction['variance_vector'])
            if "variance_of_mean_vector" in prediction:
                variance_of_mean_vectors.append(prediction["variance_of_mean_vector"])
                variance_of_variance_vectors.append(prediction["variance_of_variance_vector"])
                
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            img_paths += prediction["img_paths"]

        # glue the batches together to get one large batch
        mean_vectors = torch.cat(mean_vectors, dim=0)
        variance_vectors = torch.cat(variance_vectors, dim=0)
        if len(variance_of_mean_vectors) > 0:
            variance_of_mean_vectors = torch.cat(variance_of_mean_vectors, dim=0)
            variance_of_variance_vectors = torch.cat(variance_of_variance_vectors, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()

        # split the stuff into query and gallery

        # query feature, person ids and camera ids
        query_mean_vectors = mean_vectors[:self._num_query]
        query_variance_vectors = variance_vectors[:self._num_query]
        query_variance_of_mean_vectors = []
        query_variance_of_variance_vectors = []
        if len(variance_of_mean_vectors) > 0:
            query_variance_of_mean_vectors = variance_of_mean_vectors[:self._num_query]
            query_variance_of_variance_vectors = variance_of_variance_vectors[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]
        query_img_paths = img_paths[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_mean_vectors = mean_vectors[self._num_query:]
        gallery_variance_vectors = variance_vectors[self._num_query:]
        gallery_variance_of_mean_vectors = []
        gallery_variance_of_variance_vectors = []
        if len(variance_of_mean_vectors) > 0:
            gallery_variance_of_mean_vectors = variance_of_mean_vectors[self._num_query:]
            gallery_variance_of_variance_vectors = variance_of_variance_vectors[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]
        gallery_img_paths = img_paths[self._num_query:]

        return {
            "Q": {
                "mean_vectors": query_mean_vectors,
                "variance_vectors": query_variance_vectors,
                "variance_of_mean_vectors": query_variance_of_mean_vectors,
                "variance_of_variance_vectors": query_variance_of_variance_vectors,
                "pids": query_pids,
                "camids": query_camids,
                "img_paths": query_img_paths
            },
            "G": {
                "mean_vectors": gallery_mean_vectors,
                "variance_vectors": gallery_variance_vectors,
                "variance_of_mean_vectors": gallery_variance_of_mean_vectors,
                "variance_of_variance_vectors": gallery_variance_of_variance_vectors,
                "pids": gallery_pids,
                "camids": gallery_camids,
                "img_paths": gallery_img_paths
            },
            "Q+G": {
                "mean_vectors": mean_vectors,
                "variance_vectors": variance_vectors,
                "variance_of_mean_vectors": variance_of_mean_vectors,
                "variance_of_variance_vectors": variance_of_variance_vectors,
                "pids": pids,
                "camids": camids,
                "img_paths": img_paths
            }
        }
    
    def evaluate_embedding(self, prepared_predictions, metrics):
        
        model_best_metric_set = False

        for metric in metrics:

            self._results[metric] = OrderedDict()

            dist = build_dist(
                prepared_predictions["Q"]["mean_vectors"], 
                prepared_predictions["G"]["mean_vectors"], 
                metric, 
                query_variances=prepared_predictions["Q"]["variance_vectors"], 
                gallery_variances=prepared_predictions["G"]["variance_vectors"]
            ) # using kwargs so build_dist doesn't need a major overhaul

            if self.cfg.TEST.RERANK.ENABLED:
                logger.info("Test with rerank setting")
                k1 = self.cfg.TEST.RERANK.K1
                k2 = self.cfg.TEST.RERANK.K2
                lambda_value = self.cfg.TEST.RERANK.LAMBDA

                if self.cfg.TEST.METRIC == "cosine":
                    query_mean_vectors = F.normalize(prepared_predictions["Q"]["mean_vectors"], dim=1)
                    gallery_mean_vectors = F.normalize(prepared_predictions["G"]["mean_vectors"], dim=1)
                else:
                    query_mean_vectors = prepared_predictions["Q"]["mean_vectors"]
                    gallery_mean_vectors = prepared_predictions["G"]["mean_vectors"]
                    
                rerank_dist = build_dist(query_mean_vectors, gallery_mean_vectors, metric="jaccard", k1=k1, k2=k2)
                dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

            cmc, all_AP, all_INP = evaluate_rank(
                dist, 
                prepared_predictions["Q"]["pids"], 
                prepared_predictions["G"]["pids"], 
                prepared_predictions["Q"]["camids"], 
                prepared_predictions["G"]["camids"]
            )

            mAP = np.mean(all_AP)
            mINP = np.mean(all_INP)
            for r in [1, 5, 10]:
                self._results[metric]['Rank-{}'.format(r)] = cmc[r - 1] * 100
            self._results[metric]['mAP'] = mAP * 100
            self._results[metric]['mINP'] = mINP * 100
            self._results[metric]["metric"] = mAP * 100 # (mAP + cmc[0]) / 2 * 100 # changed this because we don't care about Rank1

            # we set this on top-level for the framework to read and decide which is model_best
            # we only set this for the first metric in the list, that is the one that decides
            if not model_best_metric_set: 
                self._results["metric"] = mAP * 100 # (mAP + cmc[0]) / 2 * 100 # this is what decides which model is treated as model_best
                model_best_metric_set = True

            if self.cfg.TEST.ROC.ENABLED:
                from .roc import evaluate_roc
                scores, labels = evaluate_roc(
                    dist, 
                    prepared_predictions["Q"]["pids"], 
                    prepared_predictions["G"]["pids"], 
                    prepared_predictions["Q"]["camids"], 
                    prepared_predictions["G"]["camids"]
                )
                fprs, tprs, thres = sklearn.metrics.roc_curve(labels, scores)

                for fpr in [1e-4, 1e-3, 1e-2]:
                    ind = np.argmin(np.abs(fprs - fpr))
                    self._results[metric]["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]
        

    def evaluate_uncertainty(self, prepared_predictions, best_metric, current_metric):
        # evaluate uncertainty values
        prepared_predictions["D1"] = {"img_paths": []}
        prepared_predictions["D2"] = {"img_paths": []}
        prepared_predictions["D3"] = {"img_paths": []}
        prepared_predictions["D4"] = {"img_paths": []}
        prepared_predictions["D"] = {"img_paths": []}

        # map uncertainty type and config option to set of vectors
        eval_set_map = {
            "MODEL": {
                "Q": prepared_predictions["Q"]["variance_of_mean_vectors"],
                "G": prepared_predictions["G"]["variance_of_mean_vectors"],
                "Q+G": prepared_predictions["Q+G"]["variance_of_mean_vectors"],
                "D": [],
                "D1": [],
                "D2": [],
                "D3": [],
                "D4": []
            },
            "DATA": {
                "Q": prepared_predictions["Q"]["variance_vectors"],
                "G": prepared_predictions["G"]["variance_vectors"],
                "Q+G": prepared_predictions["Q+G"]["variance_vectors"],
                "D": [],
                "D1": [],
                "D2": [],
                "D3": [],
                "D4": []
            },
            "DIST": {
                "Q": prepared_predictions["Q"]["variance_of_variance_vectors"],
                "G": prepared_predictions["G"]["variance_of_variance_vectors"],
                "Q+G": prepared_predictions["Q+G"]["variance_of_variance_vectors"],
                "D": [],
                "D1": [],
                "D2": [],
                "D3": [],
                "D4": []
            }
        }


        # populate distractor subsets D1...D4
        with open(os.path.join(pathlib.Path(__file__).parent, "distraction_levels.txt"), "r") as file:
            # all distractors are in gallery so we use "G" here
            shortened_prediction_paths = ["/".join(ppath.split("/")[-2:]) for ppath in prepared_predictions["G"]["img_paths"]]
            translate = {
                "MODEL": "variance_of_mean_vectors",
                "DATA": "variance_vectors", 
                "DIST": "variance_of_variance_vectors"
            }
            for line in file:
                distractor_path, distraction_level = line.strip().split(",")
                prepared_predictions["D"+distraction_level]["img_paths"].append(prepared_predictions["G"]["img_paths"][shortened_prediction_paths.index(distractor_path)])
                for uncertainty_type in ["MODEL", "DATA", "DIST"]:
                    eval_set_map[uncertainty_type]["D"+distraction_level].append( # append the uncertainty vector belonging to that uncertainty type where the original image is the distractor path from the file
                        prepared_predictions["G"][ translate[uncertainty_type] ][shortened_prediction_paths.index(distractor_path)]
                    )
        for uncertainty_type in ["MODEL", "DATA", "DIST"]:
            for set_id in ["D1", "D2", "D3", "D4"]:
                eval_set_map[uncertainty_type][set_id] = torch.stack(eval_set_map[uncertainty_type][set_id])
            eval_set_map[uncertainty_type]["D"] = torch.cat([eval_set_map[uncertainty_type][set_id] for set_id in ["D1", "D2", "D3", "D4"]])
            for set_id in ["D1", "D2", "D3", "D4"]:
                prepared_predictions["D"]["img_paths"] += prepared_predictions[set_id]["img_paths"]

        model_outputs_to_export = {
            "data": {},
            "sets": {
                "Q": [path.split("/")[-1] for path in prepared_predictions["Q"]["img_paths"]],
                "G": [path.split("/")[-1] for path in prepared_predictions["G"]["img_paths"]],
                "D1": [path.split("/")[-1] for path in prepared_predictions["D1"]["img_paths"]],
                "D2": [path.split("/")[-1] for path in prepared_predictions["D2"]["img_paths"]], 
                "D3": [path.split("/")[-1] for path in prepared_predictions["D3"]["img_paths"]],
                "D4": [path.split("/")[-1] for path in prepared_predictions["D4"]["img_paths"]]
            },
            "glossary": {
                "mean_vector": "The mean of the embedding distribution. Ordinarily, this would be the embedding vector. Computed as the mean of the multiple mean vectors that result from sampling the Bayesian module multiple times.",
                "variance_vector": "The variance of the embedding distribution. This is the estimate for data uncertainty. Computed as the mean of the multiple variance vectors that result from sampling the Bayesian module multiple times.",
                "variance_of_mean_vector": "This is the estimate for model uncertainty. We compute multiple versions of the mean vector with different weights in the Bayesian module. This is the variance over those.",
                "variance_of_variance_vector": "This is the estimate for distributional uncertainty. We compute multiple versions of the variance vector with different weights in the Bayesian module. This is the variance over those.",
                "Q": "The query subset.",
                "G": "The gallery subset.",
                "D1": "The Level-1 distractors. These are mostly images containing more than 50% of a person.",
                "D2": "The Level-2 distractors. These are mostly images containing only fragments of a person (e.g. arm, leg).",
                "D3": "The Level-3 distractors. These are mostly images of non-human objects (e.g. bikes, trees).",
                "D4": "The Level-4 distractors. These are mostly images of plain background (e.g. street, door) or unrecognizable color blobs.",
            }
        }

        for i, path in enumerate(prepared_predictions["Q+G"]["img_paths"]):
            filename = path.split("/")[-1]
            model_outputs_to_export["data"][filename] = {
                "mean_vector": [float(str(x)) for x in list(prepared_predictions["Q+G"]["mean_vectors"][i].numpy())],
                "variance_vector": [float(str(x)) for x in list(prepared_predictions["Q+G"]["variance_vectors"][i].numpy())],
                "variance_of_mean_vector": [float(str(x)) for x in list(prepared_predictions["Q+G"]["variance_of_mean_vectors"][i].numpy())],
                "variance_of_variance_vector": [float(str(x)) for x in list(prepared_predictions["Q+G"]["variance_of_variance_vectors"][i].numpy())]
            }



        # map config option to aggregation function
        AGGREGATION_MAP = {
            "min": torch.min,
            "max": torch.max,
            "avg": torch.mean
        }
        
        self._results["uncertainty"] = {}
        uncertain_images = {
            "name": "/".join(self.cfg["OUTPUT_DIR"].split("/")[-2:])
        }
        uncertainty_scores = {}

        # TODO: could do speedup by caching the metric per img by function and set

        # evaluation for each uncertainty type possible
        for uncertainty_type in ["MODEL", "DATA", "DIST"]:
            
            # extract config options
            set_ids = self.cfg.TEST.UNCERTAINTY[uncertainty_type].SETS
            aggregation_ids = self.cfg.TEST.UNCERTAINTY[uncertainty_type].AGGREGATIONS
            function_ids = self.cfg.TEST.UNCERTAINTY[uncertainty_type].FUNCTIONS
            
            # no need to create sub-dicts if we won't store anything there
            if len(set_ids) > 0 and len(aggregation_ids) > 0 and len(function_ids) > 0:
                self._results["uncertainty"][uncertainty_type] = {}

                uncertain_images[uncertainty_type] = {}
                uncertainty_scores[uncertainty_type] = {}

            # allow for specifying over which set we calculate and allow for multiple variants simultaneously
            for set_id in set_ids:

                eval_set = eval_set_map[uncertainty_type][set_id] # get set, see above

                if len(eval_set) == 0:
                    # set might be emtpy, if so: skip
                    continue

                # no need to create sub-dicts if we won't store anything there
                if len(aggregation_ids) > 0 and len(function_ids) > 0:
                    self._results["uncertainty"][uncertainty_type][set_id] = {}

                    uncertain_images[uncertainty_type][set_id] = {}
                    uncertainty_scores[uncertainty_type][set_id] = {}

                # allow for specifying how we aggregate the scores from all the inputs and allow for multiple variants simultaneously
                for aggregation_id in aggregation_ids:

                    # no need to create sub-dicts if we won't store anything there
                    if len(function_ids) > 0:
                        self._results["uncertainty"][uncertainty_type][set_id][aggregation_id] = {}
                        #uncertain_images[uncertainty_type][set_id][aggregation_id] = {}
                        #uncertainty_scores[uncertainty_type][set_id][aggregation_id] = {}
                    
                    # allow for specifying how we evaluate a score for the vectors and allow for multiple variants simultaneously
                    for function_id in function_ids:

                        # for readability in logs, add 'L' so we see it is a norm (e.g. 2 -> L2)
                        if type(function_id) != str:
                            function_label = f"L{function_id}"
                        else:
                            function_label = function_id

                        # apply the specified evaluation function to the specified set and aggregate the values using the specified aggregation function
                        # then store at the corresponding place for logging. Example log key: eval.uncertainty.DATA.Q.avg.L2
                        if uncertainty_scores[uncertainty_type][set_id].get(function_label) == None:
                            metric_per_image = get_eval_function(function_id)(eval_set) # Tensor (B)
                            # cache for other aggregation types
                            uncertainty_scores[uncertainty_type][set_id][function_label] = metric_per_image
                        else:
                            metric_per_image = uncertainty_scores[uncertainty_type][set_id][function_label]

                        # log summary metric
                        self._results["uncertainty"][uncertainty_type][set_id][aggregation_id][function_label] = AGGREGATION_MAP[aggregation_id](
                                                                                                                    metric_per_image
                                                                                                                    )

                        if current_metric > best_metric:
                            
                            # find indices that sort this list
                            metric_per_image = metric_per_image.cpu().numpy()
                            sorting_indices = np.argsort(metric_per_image)
                            
                            # get respective img_paths
                            relevant_img_paths = prepared_predictions[set_id]["img_paths"]
                            
                            # sort that
                            sorted_img_paths = [relevant_img_paths[i] for i in sorting_indices]
                            sorted_scores = [np.float64(metric_per_image[i]) for i in sorting_indices] # float32 is not json-serializabe
                            
                            # take the most and least uncertain images
                            num_uncertain_images = self.cfg.TEST.UNCERTAINTY.NUM_UNCERTAIN_IMAGES
                            most_uncertain_images = sorted_img_paths[-num_uncertain_images:] # most uncertain is last entry
                            most_uncertain_scores = sorted_scores[-num_uncertain_images:]
                            medium_uncertain_images = sorted_img_paths[len(sorted_img_paths) // 2 - num_uncertain_images // 2 : len(sorted_img_paths) // 2 - num_uncertain_images // 2 + num_uncertain_images]
                            medium_uncertain_scores = sorted_scores[len(sorted_img_paths) // 2 - num_uncertain_images // 2 : len(sorted_img_paths) // 2 - num_uncertain_images // 2 + num_uncertain_images]
                            least_uncertain_images = sorted_img_paths[:num_uncertain_images]
                            least_uncertain_scores = sorted_scores[:num_uncertain_images] # least uncertain is first entry 

                            # build dict to save them
                            uncertain_images[uncertainty_type][set_id][function_label] = {
                                "most_uncertain": most_uncertain_images,
                                "most_uncertain_scores": most_uncertain_scores,
                                "medium_uncertain": medium_uncertain_images,
                                "medium_uncertain_scores": medium_uncertain_scores,
                                "least_uncertain": least_uncertain_images,
                                "least_uncertain_scores": least_uncertain_scores
                            }                         

        if current_metric > best_metric: # passing this information to checkpointer would be more cumbersome so we save it here
            # save dict into file. could also save it to wandb but file is enough for now
            with open(os.path.join(self.cfg.OUTPUT_DIR, "uncertain_images.json"), "w") as f:
                json.dump(uncertain_images, f) 

            with open(os.path.join(self.cfg.OUTPUT_DIR, "raw_model_outputs.json"), "w") as f: # TODO: make configurable whether this is computed/saved
                json.dump(model_outputs_to_export, f) 

