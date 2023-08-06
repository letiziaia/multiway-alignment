import pandas as pd
import numpy as np
import os
from joblib import dump
from functools import partial

import multiprocessing as mp
from multiprocessing.pool import Pool

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
    _df: pd.DataFrame,
    save_to: str,
    which_score: str = "ami",
    adjusted: bool = False,
    n_tries: int = 10,
):
    """
    Generate 'n_tries' random configurations of the real data in '_df'.
    Each configuration is evaluated and the full alignment curve is dumped
    to the folder 'save_to'
    :param _df: pd.DataFrame, the original data
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
            [_df.copy()] * n_tries,
        )
        i = 0
        for value in result.get():
            dump(value, f"{save_to}/null_{i}")
            i += 1
