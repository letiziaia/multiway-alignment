import typing
import pandas as pd
import numpy as np
from itertools import combinations
from functools import partial

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

import multiprocessing as mp
from multiprocessing.pool import Pool

from src.mutual_clusters import compute_mutual_clusters
from src.mutual_clusters import get_mutual_clusters_labels

from src.common.logging import logger


def _compute_layer_expectation(
    layer: np.array, scoring_function: typing.Callable
) -> float:
    """
    :param layer: 1d np.array with clustering assignment
    :param scoring_function: partial Callable, will be called on random permutations of 'layer'
    :return: float, the expected score of 'scoring_function' under random model
    """
    _all_scores = []
    with Pool(processes=mp.cpu_count() - 1) as pool:
        result = pool.map_async(
            scoring_function, [np.random.permutation(layer) for _ in range(10)]
        )
        for value in result.get():
            _all_scores.append(value)
    return np.array(_all_scores).mean()


def compute_multilayer_alignment_score(
    cluster_labels_df: pd.DataFrame,
    mutual_clusters_labels: list,
    which_score: str = "nmi",
    adjusted: bool = False,
) -> float:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param mutual_clusters_labels: list, a list of labels for mutual clusters
    :param which_score: str, one of "nmi" or "ami"
    :param adjusted: bool, default: False
    :return: float, between 0 and 1
    """
    assert which_score in ("nmi", "ami")

    if which_score == "nmi":
        _score_f = normalized_mutual_info_score
    elif which_score == "ami":
        _score_f = adjusted_mutual_info_score

    avg_nmi = 0
    _expected_nmi = 0
    for layer_id in cluster_labels_df.columns:
        _layer = cluster_labels_df[layer_id].values
        _score = _score_f(_layer, mutual_clusters_labels, average_method="arithmetic")
        avg_nmi += _score

        if adjusted:
            _expected_nmi += _compute_layer_expectation(
                layer=_layer,
                scoring_function=partial(
                    _score_f, mutual_clusters_labels, **{"average_method": "arithmetic"}
                ),
            )
    return (avg_nmi - _expected_nmi) / len(cluster_labels_df.columns)


def compute_maximal_alignment_curve(
    cluster_labels_df: pd.DataFrame,
    which_score: str = "nmi",
    adjusted: bool = False,
) -> dict:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param which_score: str, one of "nmi" or "ami"
    :param adjusted: bool, default: False
    :return: dict[int, tuple[float, list, dict]], where the key is the size of the combination (int) and the value is a
        list, where the first element is the highest multilayer alignment score for that size,
        the second element is the list of layers that gives the highest alignment score,
        and the last element is the dictionary of mutual communities for that combination
    """
    assert which_score in ("nmi", "ami")

    best_by_combination_size = dict()
    _num_of_layers = len(cluster_labels_df.columns)
    for length in range(1, _num_of_layers + 1):
        logger.info(f"combinations of size {length}")
        # Get all combinations of cluster_labels_df.columns of length "length"
        _columns_combinations = combinations(cluster_labels_df.columns, length)

        best_layers_combination = None
        best_nmi = 0
        best_layers_combination_mutual_communities = dict()

        for l_comb in _columns_combinations:
            l_comb = list(l_comb)
            l_comb_df = cluster_labels_df[l_comb]
            mutual_clusters = compute_mutual_clusters(l_comb_df)
            mutual_clusters_labels = get_mutual_clusters_labels(mutual_clusters)
            labels_list = (
                mutual_clusters_labels.set_index("id")
                .iloc[l_comb_df.index]["label"]
                .values
            )

            # CRITERIA
            nmi = compute_multilayer_alignment_score(
                l_comb_df, labels_list, which_score=which_score, adjusted=adjusted
            )

            if nmi > best_nmi:
                best_nmi = nmi
                best_layers_combination = l_comb
                best_layers_combination_mutual_communities = mutual_clusters

        # RESULTS
        best_by_combination_size[length] = (
            best_nmi,
            best_layers_combination,
            best_layers_combination_mutual_communities,
        )
        logger.info(
            f"{length}-combination with highest score {best_nmi}: {best_layers_combination}"
        )

    return best_by_combination_size
