from itertools import combinations
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

from src.mutual_clusters import compute_mutual_clusters
from src.mutual_clusters import get_mutual_clusters_labels

from src.common.logging import logger


def compute_multilayer_alignment_score(
    cluster_labels_df: pd.DataFrame, mutual_clusters_labels: list
) -> float:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param mutual_clusters_labels: list, a list of labels for mutual clusters
    :return: float, between 0 and 1
    """
    avg_nmi = 0
    for layer_id in cluster_labels_df.columns:
        _layer = cluster_labels_df[layer_id].values
        _score = normalized_mutual_info_score(_layer, mutual_clusters_labels)
        avg_nmi += _score
    return avg_nmi/len(cluster_labels_df.columns)


def compute_maximal_alignment_curve(cluster_labels_df: pd.DataFrame) -> dict:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :return: dict[int, tuple[float, list, dict]], where the key is the size of the combination (int) and the value is a
        list, where the first element is the highest multilayer alignment score for that size,
        the second element is the list of layers that gives the highest alignment score,
        and the last element is the dictionary of mutual communities for that combination
    """
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
            nmi = compute_multilayer_alignment_score(l_comb_df, labels_list)

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
