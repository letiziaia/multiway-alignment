import pandas as pd
import matplotlib.pyplot as plt

from src.alignment_score import compute_maximal_alignment_curve


def plot_maximal_alignment_curve(
    cluster_labels_df: pd.DataFrame,
    which_score: str = "nmi",
    adjusted: bool = False,
) -> plt.Figure:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
        and column names are layers names
    :param which_score: str, one of "nmi" or "ami"
    :param adjusted: bool, default: False
    :return: plt.Figure with 2 subplots (1 row x 2 columns)
    """
    res = compute_maximal_alignment_curve(
        cluster_labels_df=cluster_labels_df, which_score=which_score, adjusted=adjusted
    )
    combination_sizes = []
    anmi_scores = []
    communities_idx = []
    communities_size = []
    for key, value in res.items():
        _anmi = value[0]
        _mc = value[2]
        combination_sizes.append(key)
        anmi_scores.append(_anmi)
        communities_idx += [key] * len(_mc)
        communities_size += [len(v) for v in _mc.values()]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(21, 10))
    ax0.plot(combination_sizes, anmi_scores, "ro--")
    ax0.set_xticks(combination_sizes)
    ax0.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax0.set_ylim(ymin=0, ymax=1.1)
    ax0.set_ylabel("maximal average NMI")
    ax0.set_xlabel("size of layers combination")
    ax1.scatter(
        communities_idx, communities_size, s=80, facecolors="none", edgecolors="g"
    )
    ax1.set_xticks(combination_sizes)
    ax1.set_yticks(range(max(communities_size) + 2, 5))
    ax1.set_ylim(ymin=0)
    ax1.set_ylabel("communities sizes")
    ax1.set_xlabel("size of layers combination")
    return fig
