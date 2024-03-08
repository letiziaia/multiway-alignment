import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
import seaborn as sns
import typing

from multilayer_alignment.alignment_score import maximal_alignment_curve
from multilayer_alignment.null_models import expected_curve_equal_sized_clusters
from multilayer_alignment.utils.logging import logger


def plot_maximal_alignment_curve(
    opinions: pd.DataFrame,
    which_score: str = "nmi",
    adjusted: bool = False,
) -> plt.Figure:
    """
    :param opinions: pd.DataFrame having one column per layer and one row per node,
        where each element a_ij is an integer representing the cluster labels for node i at layer j
        and column names are layers names
    :param which_score: str, one of "nmi" or "ami"
    :param adjusted: bool, default: False
    :return: plt.Figure with 2 subplots (1 row x 2 columns)
    """
    _, res = maximal_alignment_curve(
        opinions=opinions, which_score=which_score, adjusted=adjusted
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
    opinions: typing.Optional[pd.DataFrame],
    which_score: typing.Optional[str],
    adjusted: typing.Optional[bool],
    full_dump_path: typing.Optional[str],
) -> plt.Figure:
    """
    :param opinions: pd.DataFrame having one column per layer and one row per node,
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
            opinions is not None and which_score is not None and adjusted is not None
        ), "not enough inputs"
        all_results, _ = maximal_alignment_curve(
            opinions=opinions,
            which_score=which_score,
            adjusted=adjusted,
        )

    _points = [
        (k.split("+")[0], v[0], k.split("+")[1:]) for k, v in all_results.items()
    ]
    points = pd.DataFrame(_points)
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
        text = [lab.replace("_", " ") for lab in label]
        text = "\n".join(text)
        plt.annotate(
            text,
            (x_top[i], y_top[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            fontsize=11,
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


def plot_full_alignment_with_null_models(
    full_result: str, full_null_path: str
) -> plt.Figure:
    r = load(full_result)
    points = [(k.split("+")[0], v[0], k.split("+")[1:]) for k, v in r.items()]
    points = pd.DataFrame(points)

    points_df = []
    for i in range(10):
        r = load(f"{full_null_path}/null_{i}")
        _points = [(k.split("+")[0], v[0], k.split("+")[1:]) for k, v in r.items()]
        points_df.append(pd.DataFrame(_points))

    points_df = pd.concat(points_df, ignore_index=True)

    top = points.sort_values(by=[1, 0], ascending=False).groupby(0).head(1)
    x_top = top[0]
    x_top = [int(v) for v in x_top]
    y_top = top[1]

    _strip = points[~points.index.isin(top.index)]

    null_avg = points_df.groupby(0)[1].mean().to_dict()
    y_top_ = y_top.values - np.array([null_avg[k] for k in null_avg.keys()])

    null_upper_q = points_df.groupby(0)[1].quantile(q=0.975).to_dict()
    null_lower_q = points_df.groupby(0)[1].quantile(q=0.025).to_dict()
    sig = [
        (_i, v - null_avg[_i])
        for _i, v in _strip[[0, 1]].values
        if v > null_upper_q[_i] or v < null_lower_q[_i]
    ]
    not_sig = [
        (_i, v - null_avg[_i])
        for _i, v in _strip[[0, 1]].values
        if null_lower_q[_i] <= v <= null_upper_q[_i]
    ]

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    maxim = ax.scatter(
        x_top,
        y_top_,
        marker="s",
        linestyle=":",
        color="blue",
        alpha=1.0,
    )
    # Annotate each point with its label
    for j, (i, label) in enumerate(top[[2]].iterrows()):
        label = label[2]
        text = [lab.replace("_", " ") for lab in label]
        text = "\n".join(text)
        ax.annotate(
            text,
            (x_top[j], y_top_[j]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            fontsize=14,
        )
    _sig = pd.DataFrame(sig)
    _sig["jit"] = [
        int(v) + (random.random() / 10) * ((-1) ** (i + 1))
        for i, v in enumerate(_sig[0].values)
    ]
    _not_sig = pd.DataFrame(not_sig)
    _not_sig["jit"] = [
        int(v) + (random.random() / 10) * ((-1) ** (i + 1))
        for i, v in enumerate(_not_sig[0].values)
    ]
    first = ax.scatter(
        _sig["jit"], _sig[1], marker="o", color="red", alpha=0.7, label="significant"
    )
    second = ax.scatter(
        _not_sig["jit"],
        _not_sig[1],
        marker=markers.MarkerStyle(marker="o", fillstyle="none"),
        color="blueviolet",
        alpha=0.7,
        label="explained by null model",
    )

    # Remove the left and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # change the fontsize
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    plt.xlabel("number of layers", fontsize=18)
    plt.ylabel(r"$A - <A_{null}>$", fontsize=18)

    # Create an inset in the lower right corner (loc=4) with borderpad=1, i.e.
    # borderpad 1 = 10 points padding (as 10pt is the default fontsize) to the parent axes
    axins = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc=4,
        borderpad=4,
    )
    axins.plot(
        x_top,
        y_top,
        marker="s",
        linestyle=":",
        color="blue",
        alpha=1.0,
    )
    sns.lineplot(
        data=points_df,
        x=points_df[0].astype(int),
        y=1,
        estimator="mean",
        errorbar=("ci", 95),
        n_boot=1000,
        color="black",
        markers="o",
        label=r"null model (95$\%$ c.i.)",
        ax=axins,
    )
    sns.lineplot(
        x=x_top,
        y=expected_curve_equal_sized_clusters(int(max(int(c) for c in x_top))),
        linestyle="-.",
        color="grey",
        label="equal-sized clusters",
    )
    axins.set_xlabel("number of layers")
    axins.set_ylabel(r"$A$")
    axins.set_ylim(0, 1.0)
    axins.set_aspect(1.0 / axins.get_data_ratio(), adjustable="box")

    ax.legend(
        (maxim, first, second),
        ("maximum", "significant", "explained by null model"),
        loc="center right",
        fontsize="12",
    )
    return fig
