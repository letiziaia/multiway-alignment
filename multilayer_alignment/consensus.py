import pandas as pd
from typing import Any, Dict, Set, Union


def get_consensus_labels(consensus_partition: Dict[str, Set[Any]]) -> pd.DataFrame:
    """
    :param consensus_partition: a dictionary of consensus group label (str) -> consesus group members (set)
    :return: pd.DataFrame with column 'id' for the element id and column 'label' for the element label
    """
    nodes_id = []
    labels = []
    for k, v in consensus_partition.items():
        for elm in v:
            nodes_id.append(elm)
            labels.append(k)
    return pd.DataFrame({"id": nodes_id, "label": labels}).sort_values(by="id")


def get_consensus_partition(
    opinions: Union[pd.DataFrame, pd.Series[Any]]
) -> Dict[str, Set[Any]]:
    """
    Returns the consensus groups (faster)
    :param opinions: pd.DataFrame having one column per topic and one row per individual,
        where each element a_ij represents the opinion for individual i on topic j
        and columns names are the topic names
    :return: dict[str, set], a dictionary of consensus group label (str) -> consensus group members (set)
        Note: Only non-empty sets are returned!
    ------------
    Example
    ------------
    E.g.:
        opinions:               A | B | C
                                ---------
                             0  0   1   0
                             1  0   1   0
                             2  1   0   1
        consensus groups:
            A0_B0_C0 -> {}, A0_B0_C1 -> {}, A0_B1_C0 -> {0, 1}, A0_B1_C1 -> {},
            A1_B0_C0 -> {}, A1_B0_C1 -> {2}, A1_B1_C0 -> {}, A1_B1_C1 -> {}
    >>> df = pd.DataFrame({"A": [0, 0, 1], "B": [1, 1, 0], "C": [0, 0, 1]})
    >>> get_consensus_partition(opinions=df)
    """
    consensus_groups = {}
    _topics = list(opinions.columns)
    _cg = opinions.groupby(by=_topics).groups
    for key, value in _cg.items():
        if len(_topics) == 1:
            _formatted_key = f"{_topics[0]}{key}"
        else:
            _joined_key = [
                "".join((str(col_name), str(label)))
                for col_name, label in zip(_topics, key)
            ]
            _formatted_key = "_".join(_joined_key)
        consensus_groups[_formatted_key] = set(value)
    return consensus_groups


def get_consensus_partition_recursive(
    opinions: Union[pd.DataFrame, pd.Series[Any]],
    consensus_groups: Dict[str, Set[Any]] = {},
    next_topic_idx: int = 0,
) -> Dict[str, Set[Any]]:
    """
    Recursive function that traverses all the topics and builds the consensus groups
    :param opinions: pd.DataFrame having one column per topic and one row per individual,
        where each element a_ij represents the opinion of individual i on topic j
    :param consensus_groups: dict[str, set], a dictionary of consensus group label (str) -> consensus group members (set)
        The consensus group labels are built at each step by combining the labels of the groups that are intersected.
        Default: empty dictionary
    :param next_topic_idx: int, the index of next topic to consider
        Default: 0 (first column in opinions)
    :return: current_sets, dict[str, set], a dictionary of consensus group label (str) -> consensus group members (set)
        Note: Only non-empty sets are returned!
    ------------
    Example
    ------------
    E.g.:
        opinions:               A | B | C
                                ---------
                             0  0   1   0
                             1  0   1   0
                             2  1   0   1
        consensus groups (first iteration):
            A0_B0 -> {}, A0_B1 -> {0, 1}, A1_B0 -> {2}, A1_B1 -> {}
        consensus groups (final iteration):
            A0_B0_C0 -> {}, A0_B0_C1 -> {}, A0_B1_C0 -> {0, 1}, A0_B1_C1 -> {},
            A1_B0_C0 -> {}, A1_B0_C1 -> {2}, A1_B1_C0 -> {}, A1_B1_C1 -> {}
    >>> df = pd.DataFrame({"A": [0, 0, 1], "B": [1, 1, 0], "C": [0, 0, 1]})
    >>> get_consensus_partition_recursive(opinions=df)
    """
    _num_of_layers = len(opinions.columns)
    # recursion base case: no layer left to be processed
    if next_topic_idx == _num_of_layers:
        return consensus_groups
    else:
        _next_layer = opinions.columns[next_topic_idx]
        _updated_mutual_clusters = {}
        if len(consensus_groups) == 0:
            # in this case, we need to create the keys for the dictionary from scratch
            for _cluster_label in opinions[_next_layer].unique():
                _cluster_content = opinions[opinions[_next_layer] == _cluster_label][
                    _next_layer
                ].index
                _key = str(_next_layer) + str(_cluster_label)
                _updated_mutual_clusters[_key] = set(_cluster_content)
            return get_consensus_partition_recursive(
                opinions, _updated_mutual_clusters, next_topic_idx + 1
            )
        else:
            # in this case, we start from the current mutual clusters and compare with the current layer
            _updated_mutual_clusters = {}
            for mc_id, curr_mc in consensus_groups.items():
                for _cluster_label in opinions[_next_layer].unique():
                    _cluster_content = opinions[
                        opinions[_next_layer] == _cluster_label
                    ][_next_layer].index
                    _new_key_suffix = str(_next_layer) + str(_cluster_label)
                    _key = str(mc_id) + "_" + _new_key_suffix
                    _new_set = curr_mc.intersection(set(_cluster_content))
                    if len(_new_set) > 0:
                        _updated_mutual_clusters[_key] = _new_set
            return get_consensus_partition_recursive(
                opinions, _updated_mutual_clusters, next_topic_idx + 1
            )
