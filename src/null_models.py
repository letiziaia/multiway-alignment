import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
from itertools import combinations
from joblib import dump
from functools import partial
import typing

import multiprocessing as mp
from multiprocessing.pool import Pool
from tqdm import tqdm

from src.alignment_score import compute_maximal_alignment_curve

from src.common.logging import logger


def get_null_model(cluster_labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :return: pd.DataFrame, having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    """
    null = pd.DataFrame()
    for layer_id in cluster_labels_df.columns:
        _layer = cluster_labels_df[layer_id].fillna(9).values
        null[layer_id] = np.random.permutation(_layer)

    return null


def _one_iter(
    cluster_labels_df: pd.DataFrame, which_score: str = "ami", adjusted: bool = False
) -> dict:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param which_score: str
    :param adjusted: bool
    :return: dict
    """
    null = get_null_model(cluster_labels_df=cluster_labels_df)

    _full_res, _ = compute_maximal_alignment_curve(
        null, which_score=which_score, adjusted=adjusted
    )
    return _full_res


def random_full_alignment_curves(
    df: pd.DataFrame,
    save_to: str,
    which_score: str = "ami",
    adjusted: bool = False,
    n_tries: int = 10,
):
    """
    Generate 'n_tries' random configurations of the real data in 'df'.
    Each configuration is evaluated and the full alignment curve is dumped
    to the folder 'save_to'
    :param df: pd.DataFrame, the original data
    :param save_to: str, name of the folder
    :param which_score: str, the score to use
        Default: "ami"
    :param adjusted: bool
        Default: False
    :param n_tries: int, name of random configurations to generate
        Default: 10
    :return: None
    """
    if not os.path.exists(save_to):
        os.makedirs(save_to)
        logger.info(f"Created new directory {save_to}")
    with Pool(processes=mp.cpu_count() - 1) as pool:
        result = pool.map_async(
            partial(_one_iter, **{"which_score": which_score, "adjusted": adjusted}),
            [df.copy()] * n_tries,
        )
        i = 0
        for value in result.get():
            dump(value, f"{save_to}/null_{i}")
            i += 1


def expected_curve(cluster_labels_df: pd.DataFrame) -> typing.List:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :return: list of expected scores based on average NMI (normalized by arithmetic average)
    """
    _expected_best_scores = []
    _layers = list(cluster_labels_df.columns)

    _a = cluster_labels_df.copy()
    _a["count"] = 1

    for length in range(2, len(_layers) + 1):
        logger.info(f"combinations of size {length}")
        # Get all combinations of cluster_labels_df.columns of length "length"
        _columns_combinations = combinations(_layers, length)

        best_score = 0

        for l_comb in tqdm(_columns_combinations):
            l_mc = []
            mc_pk = (_a.groupby(list(l_comb))["count"].count() / len(_a)).values
            h_mc = entropy(mc_pk)
            for lay in l_comb:
                l_pk = (_a.groupby(lay)["count"].count() / len(_a)).values
                _ratio = h_mc / entropy(l_pk)
                l_mc.append(1 / (1 + _ratio))

            score = sum(l_mc) * 2 / length

            if score > best_score:
                best_score = score

        _expected_best_scores.append(best_score)
    return _expected_best_scores


def expected_curve_equal_sized_clusters(n_layers: int) -> typing.List:
    return [2 / (1 + k) for k in range(2, n_layers + 1)]