# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Modified by Andreas Gebhardt in 2024.
"""

import logging
import pprint
import sys
from collections import Mapping, OrderedDict

import numpy as np
from tabulate import tabulate
from termcolor import colored


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.
    Args:
        results (OrderedDict): {metric -> score}
    """
    # unordered results cannot be properly printed
    assert isinstance(results, OrderedDict) or not len(results), results
    logger = logging.getLogger(__name__)

    dataset_name = results.pop('dataset')
    metrics = ["Dataset"] + [k for k in results]
    csv_results = [(dataset_name, *list(results.values()))]

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=metrics,
        numalign="left",
    )

    logger.info("Evaluation results in csv format: \n" + colored(table, "cyan"))


def print_csv_format_multi(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.

    This is compatible with the Uncertainty framework and allows for multiple distance measures.

    Args:
        results (OrderedDict): {distance -> (OrderedDict): {metric -> score}}
    """
    # unordered results cannot be properly printed
    assert isinstance(results, OrderedDict) or not len(results), results
    logger = logging.getLogger(__name__)

    # remove metric used for model_best determination
    metric = results.pop("metric")

    dataset_name = results.pop('dataset')

    uncertainty_metrics = results.pop("uncertainty")
    
    metrics = ["Dataset", "Distance"] + [k for k in results[next(iter(results))]] # just use the metrics from the first entry as they should be the same in each anyway

    csv_results = []

    for distance_name, metric_values in results.items():
        # unordered results cannot be properly printed
        assert isinstance(results[distance_name], OrderedDict) or not len(results[distance_name]), results[distance_name]
        csv_results.append((dataset_name, distance_name, *list(metric_values.values()))) # construct a line of csv

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=metrics,
        numalign="left",
    )

    logger.info("Evaluation results in markdown format: \n" + colored(table, "cyan"))


    # same for uncertainty stuff

    column_headers = ["Dataset", "Type", "Set", "Agg.", "Function", "Value"]

    csv_results_uncertainty = []

    for uncertainty_type in uncertainty_metrics:
        for eval_set in uncertainty_metrics[uncertainty_type]:
            for aggregation in uncertainty_metrics[uncertainty_type][eval_set]:
                for eval_function in uncertainty_metrics[uncertainty_type][eval_set][aggregation]:
                    csv_results_uncertainty.append((dataset_name, uncertainty_type, eval_set, aggregation, eval_function, uncertainty_metrics[uncertainty_type][eval_set][aggregation][eval_function]))

    # tabulate it
    table_uncertainy = tabulate(
        csv_results_uncertainty,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=column_headers,
        numalign="left",
    )

    logger.info("Evaluation results for uncertainty values in markdown format: \n" + colored(table_uncertainy, "cyan"))

    results["uncertainty"] = uncertainty_metrics # put back for wandb

    results["metric"] = metric # need to put it back because the next hook (checkpointer) still needs it.


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task][metric]
        if not np.isfinite(actual):
            ok = False
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.
    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
