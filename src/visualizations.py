import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import typing

from src.alignment_score import compute_maximal_alignment_curve
from src.common.logging import logger


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
    _, res = compute_maximal_alignment_curve(
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


def plot_full_alignment_curve(
    cluster_labels_df: typing.Optional[pd.DataFrame],
    which_score: typing.Optional[str],
    adjusted: typing.Optional[bool],
    full_dump_path: typing.Optional[str],
) -> plt.Figure:
    """
    :param cluster_labels_df: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
        and column names are layers names
    :param which_score: str, one of "nmi" or "ami" or None
    :param adjusted: bool
    :param full_dump_path: str
    :return: plt.Figure
    """
    if full_dump_path is not None:
        all_results = load(full_dump_path)
    else:
        assert (
            cluster_labels_df is not None
            and which_score is not None
            and adjusted is not None
        ), "not enough inputs"
        all_results, _ = compute_maximal_alignment_curve(
            cluster_labels_df=cluster_labels_df,
            which_score=which_score,
            adjusted=adjusted,
        )

    points = [(k.split("+")[0], v[0], k.split("+")[1:]) for k, v in all_results.items()]
    points = pd.DataFrame(points)
    top = points.sort_values(by=[1, 0], ascending=False).groupby(0).head(1)

    logger.info(f"Area under the curve: {np.trapz(top[1], dx=1 / (len(top) - 1))}")

    x = points[~points.index.isin(top.index)][0]
    y = points[~points.index.isin(top.index)][1]

    x_top = top[0]
    y_top = top[1]

    fig = plt.figure(figsize=(20, 10))
    marker = markers.MarkerStyle(marker="s", fillstyle="none")
    plt.plot(
        x_top,
        y_top,
        marker=marker,
        linestyle=":",
        color="blue",
        alpha=0.7,
    )
    # Annotate each point with its label
    for i, label in top[[2]].iterrows():
        label = label[2]
        text = [l.replace("_", " ") for l in label]
        text = "\n".join(text)
        plt.annotate(
            text,
            (x_top[i], y_top[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            fontsize=11,
            # bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5)
        )
    plt.scatter(
        x,
        y,
        marker="x",
        color="blueviolet",
        alpha=0.7,
    )
    plt.xticks(range(2, int(max(y) + 1)))

    # Remove the left and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.xlabel("number of layers")
    plt.ylabel("maximal alignment score")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    return fig
