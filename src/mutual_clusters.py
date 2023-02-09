import pandas as pd


def get_mutual_clusters_labels(mutual_clusters: dict) -> pd.DataFrame:
    """
    :param mutual_clusters: a dictionary of mutual cluster label (str) -> mutual cluster members (set)
    :return: pd.DataFrame with column 'id' for the element id and column 'label' for the element label
    """
    nodes_id = []
    labels = []
    for k, v in mutual_clusters.items():
        for elm in v:
            nodes_id.append(elm)
            labels.append(k)
    return pd.DataFrame({"id": nodes_id, "label": labels})


def compute_mutual_clusters(
    cluster_labels_df: pd.DataFrame
) -> dict:
    """
    Returns the mutual clusters (faster)
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
        and columns names are the layers names
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
        mutual_clusters:
            A0_B0_C0 -> {}, A0_B0_C1 -> {}, A0_B1_C0 -> {0, 1}, A0_B1_C1 -> {},
            A1_B0_C0 -> {}, A1_B0_C1 -> {2}, A1_B1_C0 -> {}, A1_B1_C1 -> {}
    >>> df = pd.DataFrame({"A": [0, 0, 1], "B": [1, 1, 0], "C": [0, 0, 1]})
    >>> compute_mutual_clusters(cluster_labels_df=df)
    """
    mutual_clusters = {}
    _layers = list(cluster_labels_df.columns)
    _mc = cluster_labels_df.groupby(by=_layers).groups
    for key, value in _mc.items():
        if len(_layers) == 1:
            _formatted_key = f"{_layers[0]}{key}"
        else:
            _joined_key = ["".join((str(col_name), str(label))) for col_name, label in zip(_layers, key)]
            _formatted_key = "_".join(_joined_key)
        mutual_clusters[_formatted_key] = set(value)
    return mutual_clusters


def compute_mutual_clusters_recursive(
    cluster_labels_df: pd.DataFrame, mutual_clusters: dict = {}, next_layer_idx: int = 0
) -> dict:
    """
    Recursive function that traverses all the layers and builds the mutual clusters
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
    :param mutual_clusters: dict[str, set], a dictionary of mutual cluster label (str) -> mutual cluster members (set)
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
    >>> compute_mutual_clusters_recursive(cluster_labels_df=df)
    """
    _num_of_layers = len(cluster_labels_df.columns)
    # recursion base case: no layer left to be processed
    if next_layer_idx == _num_of_layers:
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
            return compute_mutual_clusters_recursive(
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
            return compute_mutual_clusters_recursive(
                cluster_labels_df, _updated_mutual_clusters, next_layer_idx + 1
            )
