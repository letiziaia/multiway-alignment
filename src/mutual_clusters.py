import pandas as pd


def compute_mutual_clusters(
    cluster_labels_df: pd.DataFrame, mutual_clusters: dict = {}, next_layer_idx: int = 0
) -> dict:
    """
    Recursive function that traverses all the layers and builds the mutual clusters
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param mutual_clusters: dict[str, set], a dictionary of mutual cluaster label (str) -> mutual cluster members (set)
        The mutual cluster labels are built at each step by combining the labels of the clusters that are intersected.
        Default: empty dictionary
    :param next_layer_idx: int, the index of next layer to consider
        Default: 0 (first column in cluster_labels_df)
    :return: current_sets, dict[str, set], a dictionary of mutual cluster label (str) -> mutual cluster members (set)
        Note: Only non-empty sets are returned!
    ------------
    Example
    ------------
    E.g.:
        cluster_labels_df:      A | B | C
                                ---------
                             0  0   1   0
                             1  0   1   0
                             2  1   0   1
        mutual_clusters (first iteration):
            A0_B0 -> {}, A0_B1 -> {0, 1}, A1_B0 -> {2}, A1_B1 -> {}
        mutual_clusters (final iteration):
            A0_B0_C0 -> {}, A0_B0_C1 -> {}, A0_B1_C0 -> {0, 1}, A0_B1_C1 -> {},
            A1_B0_C0 -> {}, A1_B0_C1 -> {2}, A1_B1_C0 -> {}, A1_B1_C1 -> {}
    >>> df = pd.DataFrame({"A": [0, 0, 1], "B": [1, 1, 0], "C": [0, 0, 1]})
    >>> compute_mutual_clusters(cluster_labels_df=df)
    """
    _num_of_layers = len(cluster_labels_df.columns)
    # recursion base case: no layer left to be processed
    if next_layer_idx == _num_of_layers:
        # # TODO: is the chunck below ever needed?
        # # at this point, we have lost all the nodes that do not belong to any mutual cluster
        # # therefore, we need to add them back as singletons
        # _updated_mutual_clusters = {}
        # _all_nodes = list(cluster_labels_df.index)
        # for n in _all_nodes:
        #     _found = False
        #     _singleton_key = "s" + str(n)
        #     for mc in mutual_clusters.values():
        #         if n in mc:
        #             _found = True
        #     if not _found:
        #         _updated_mutual_clusters[_singleton_key] = {n}
        # _updated_mutual_clusters.update(mutual_clusters)
        # return _updated_mutual_clusters
        return mutual_clusters
    else:
        _next_layer = cluster_labels_df.columns[next_layer_idx]
        _updated_mutual_clusters = {}
        if len(mutual_clusters) == 0:
            # in this case, we need to create the keys for the dictionary from scratch
            for _cluster_label in cluster_labels_df[_next_layer].unique():
                _cluster_content = cluster_labels_df[
                    cluster_labels_df[_next_layer] == _cluster_label
                ][_next_layer].index
                _key = str(_next_layer) + str(_cluster_label)
                _updated_mutual_clusters[_key] = set(_cluster_content)
            return compute_mutual_clusters(
                cluster_labels_df, _updated_mutual_clusters, next_layer_idx + 1
            )
        else:
            # in this case, we start from the current mutual clusters and compare with the current layer
            _updated_mutual_clusters = {}
            for mc_id, curr_mc in mutual_clusters.items():
                for _cluster_label in cluster_labels_df[_next_layer].unique():
                    _cluster_content = cluster_labels_df[
                        cluster_labels_df[_next_layer] == _cluster_label
                    ][_next_layer].index
                    _new_key_suffix = str(_next_layer) + str(_cluster_label)
                    _key = str(mc_id) + "_" + _new_key_suffix
                    _new_set = curr_mc.intersection(set(_cluster_content))
                    if len(_new_set) > 0:
                        _updated_mutual_clusters[_key] = _new_set
            return compute_mutual_clusters(
                cluster_labels_df, _updated_mutual_clusters, next_layer_idx + 1
            )
