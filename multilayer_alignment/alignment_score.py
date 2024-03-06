import typing
import pandas as pd
import numpy as np
from itertools import combinations
from functools import partial
from joblib import dump

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

import multiprocessing as mp
from multiprocessing.pool import Pool
from tqdm import tqdm

from multilayer_alignment.consensus import get_consensus_partition
from multilayer_alignment.consensus import get_consensus_labels

from multilayer_alignment.utils.logging import logger


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
            # NOTE: assuming scores are always >=0
            _all_scores.append(max(value, 0))
    return np.array(_all_scores).mean()


def compute_multilayer_alignment_score(
    cluster_labels_df: pd.DataFrame,
    mutual_clusters_labels: typing.List,
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
    _expected_nmi = 0.0
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
    dump_to: typing.Optional[str] = None,
) -> typing.Tuple:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param which_score: str, one of "nmi" or "ami"
    :param adjusted: bool, default: False
    :param dump_to: Optional[str], filename to save results
        Default: None
    :return: Tuple[dict[str, tuple[float, list]], dict[int, tuple[float, list, dict]]],
        a tuple of two dictionaries, the first one including all the scores for all the combinations,
        and the second one being the maximal alignment curve.
        In the first dictionary, the key is the size of the combination (int) and
        the list of layers and the value is a tuple, where the first element is the multilayer alignment score,
        and the second element is the dictionary of mutual communities for that combination;
        In the second dictionary, the key is the size of the combination (int) and
        the value is a list, where the first element is the highest multilayer alignment score for that size,
        the second element is the list of layers that gives the highest alignment score,
        and the last element is the dictionary of mutual communities for that combination
    """
    assert which_score in ("nmi", "ami")

    best_by_combination_size = dict()
    all_scores_by_combination_size = dict()
    _num_of_layers = len(cluster_labels_df.columns)
    # skipping size 1
    for length in range(2, _num_of_layers + 1):
        logger.info(f"combinations of size {length}")
        # Get all combinations of cluster_labels_df.columns of length "length"
        _columns_combinations = combinations(cluster_labels_df.columns, length)

        best_layers_combination = None
        best_nmi = 0
        # best_layers_combination_mutual_communities = dict()

        for l_comb in tqdm(_columns_combinations):
            l_comb = list(l_comb)
            l_comb_df = cluster_labels_df[l_comb].copy()
            # keep only items that have labels for all items in l_comb and reindex
            l_comb_df.dropna(inplace=True)
            l_comb_df.reset_index(drop=True, inplace=True)
            mutual_clusters = get_consensus_partition(l_comb_df)
            mutual_clusters_labels = get_consensus_labels(mutual_clusters)
            labels_list = (
                mutual_clusters_labels.set_index("id")
                .iloc[l_comb_df.index]["label"]
                .values
            )

            # CRITERIA
            nmi = compute_multilayer_alignment_score(
                l_comb_df, labels_list, which_score=which_score, adjusted=adjusted
            )

            all_scores_by_combination_size[f"{length}+" + "+".join(sorted(l_comb))] = (
                nmi
            )
            # (
            # nmi,
            # mutual_clusters,
            # )

            if nmi > best_nmi:
                best_nmi = nmi
                best_layers_combination = l_comb
                # best_layers_combination_mutual_communities = mutual_clusters

        # RESULTS
        best_by_combination_size[length] = (
            best_nmi,
            best_layers_combination,
            # best_layers_combination_mutual_communities,
        )
        logger.info(
            f"{length}-combination with highest score {best_nmi}: {best_layers_combination}"
        )

    if dump_to:
        dump(all_scores_by_combination_size, dump_to + "_all")
        dump(best_by_combination_size, dump_to + "_best")

    return all_scores_by_combination_size, best_by_combination_size


# def reduced_mutual_info_score(labels_true: np.array, labels_pred: np.array, **kwargs) -> float:
#     """
#     :param labels_true: np.array of clusters labels
#     :param labels_pred: np.array of clusters labels
#     :param kwargs: keywords arguments (placeholder, not used)
#     :return: float, reduced (normalized) mutual information score
#     """
#     # c1 = Clustering(elm2clu_dict={i: [l] for i, l in enumerate(labels_true)})
#     # c2 = Clustering(elm2clu_dict={i: [l] for i, l in enumerate(labels_pred)})
#     print(labels_true)
#     print(labels_pred)
#     c1 = Clustering().from_membership_list(labels_true)
#     c2 = Clustering().from_membership_list(labels_pred)
#     bits = len(np.unique(labels_true))
#     print(bits)
#     print(rmi(c1, c2, norm_type="normalized", logbase=bits))
#     print(rmi(c1, c2, norm_type="none", logbase=bits))
#     input()
#     return max(0, rmi(c1, c2, norm_type="normalized", logbase=bits))
