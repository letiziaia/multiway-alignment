import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from multilayer_alignment.utils.logging import logger  # type: ignore


exclude_independent = True

start_at = 97
end_at = 113  # included-

POLICIES = [
    "Macroeconomics",
    "CivilRights",
    "Health",
    "Agriculture",
    "LaborAndEmployment",
    "Education",
    "Environment",
    "Energy",
    "Immigration",
    "Transportation",
    "Law",
    "SocialWelfare",
    "CommunityAndHousing",
    "Finance",
    "Defense",
    "Science",
    "ForeignTrade",
    "InternationalAffairs",
    "Government",
    "PublicLandsAndWater",
    "Other",
]
POLICIES = sorted(POLICIES)


topics_dict = {
    1: "Macroeconomics",
    2: "CivilRights",
    3: "Health",
    4: "Agriculture",
    5: "LaborAndEmployment",
    6: "Education",
    7: "Environment",
    8: "Energy",
    9: "Immigration",
    10: "Transportation",
    # 11: '',
    12: "Law",
    13: "SocialWelfare",
    14: "CommunityAndHousing",
    15: "Finance",
    16: "Defense",
    17: "Science",
    18: "ForeignTrade",
    19: "InternationalAffairs",
    20: "Government",
    21: "PublicLandsAndWater",
    99: "Other",
}


def get_rollcalls(starting_from_congress=start_at, ending_at_congress=end_at):
    df = pd.read_json("us_data/rollcall.json")
    df.dropna(subset=["bill_number"], inplace=True)
    df["congress"] = df["congress"].astype(str)
    df["cong_bill_number"] = df[["congress", "bill_number"]].apply("".join, axis=1)
    df["congress"] = df["congress"].astype(int)
    df = df[
        (df["congress"] >= starting_from_congress)
        & (df["congress"] <= ending_at_congress)
    ].copy()

    cmp = pd.read_csv(
        "us_data/bills_comparative.csv", usecols=["bill_id", "cong", "majortopic"]
    )
    cmp["cong"] = cmp["cong"].astype(int)
    cmp["bill_id"] = cmp["bill_id"].str.replace("-", "")
    cmp = cmp[
        (cmp["cong"] >= starting_from_congress) & (cmp["cong"] <= ending_at_congress)
    ].copy()
    cmp.drop(columns=["cong"], inplace=True)
    cmp.dropna(subset=["majortopic"], inplace=True)
    cmp["majortopic"] = cmp["majortopic"].astype(int)

    df = pd.merge(df, cmp, how="inner", left_on="cong_bill_number", right_on="bill_id")
    df["policy"] = [topics_dict[int(x)] for x in df["majortopic"].values]
    # exclude "other"
    df = df[df["policy"] != "Other"].copy()

    # get year
    df["year"] = pd.to_datetime(df["date"]).dt.year

    df["voting_members"] = df[["yea_count", "nay_count"]].sum(axis=1)
    return df[
        [
            "year",
            "congress",
            "rollnumber",
            "bill_number",
            "voting_members",
            "yea_count",
            "nay_count",
            "chamber",
            "policy",
        ]
    ].copy()


all_congresses = get_rollcalls()
mapping = all_congresses[["year", "congress"]].drop_duplicates()
YEARS = sorted(all_congresses["year"].astype(int).unique())


def get_votes_by_chamber_and_year(chamber, year):
    all_votes = get_rollcalls()
    all_votes = all_votes[all_votes["policy"].isin(POLICIES)].copy()
    return all_votes[
        (all_votes["chamber"] == chamber) & (all_votes["year"].astype(int) == int(year))
    ].copy()


def get_members_by_chamber_and_year(chamber, year):
    congress = mapping[mapping["year"].astype(int) == int(year)]["congress"].unique()
    if len(congress) > 0:
        congress = congress[0]

    all_members = pd.read_csv("us_data/HSall_members.csv")[
        ["congress", "chamber", "icpsr", "party_code"]
    ]

    if exclude_independent:
        all_members = all_members[all_members["party_code"] != 328]

    all_members["congress"] = all_members["congress"].astype(int)
    all_members["icpsr"] = all_members["icpsr"].astype(int)
    return all_members[
        (all_members["chamber"] == chamber) & (all_members["congress"] == congress)
    ].copy()


def get_individual_votes_by_chamber_and_year(senate_or_house, year):
    votes_w_topics = get_votes_by_chamber_and_year(senate_or_house, year)
    rollcalls = list(votes_w_topics["rollnumber"].astype(int).unique())

    df = pd.read_csv("us_data/HSall_votes.csv").drop(columns=["prob"])
    df["congress"] = df["congress"].astype(int)
    df["icpsr"] = df["icpsr"].astype(int)

    bins = pd.IntervalIndex.from_tuples([(0, 0), (0, 3), (3, 6), (6, 9)])
    df["vote"] = pd.cut(df["cast_code"].values, bins=bins)
    df["vote"] = (
        df["vote"]
        .astype(str)
        .replace(
            to_replace={
                "(0, 0]": "discard",
                "(0, 3]": "y",
                "(3, 6]": "n",
                "(6, 9]": "o",
            }
        )
    )

    df = df[
        (df["rollnumber"].astype(int).isin(rollcalls))
        & (df["chamber"] == senate_or_house)
    ].copy()

    ind_votes_w_partycodes = pd.merge(
        df,
        get_members_by_chamber_and_year(senate_or_house, year),
        how="inner",
        on=["congress", "chamber", "icpsr"],
    )

    ind_votes_w_partycodes_and_topic = pd.merge(
        ind_votes_w_partycodes,
        votes_w_topics[["congress", "chamber", "rollnumber", "policy"]],
        how="inner",
        on=["congress", "chamber", "rollnumber"],
    ).sort_values(by=["rollnumber", "icpsr"])
    return ind_votes_w_partycodes_and_topic


def get_similarity(data, between, how):
    """
    :param: between, str, "mps" or "votes"
    :param: how, str, "nmi" for normalized mutual information,
                      "cosine" for cosine similarity,
                      "angular" for angular similarity
    :return: square pandas DataFrame
    """

    if between == "mps":
        data = data.T
    elif between == "votes":
        data = data
    else:
        logger.warning('Choose either "mps" or "votes".')
        return

    header = list(data.columns)
    if how == "nmi":
        NMI = pdist(data.values.T, lambda u, v: normalized_mutual_info_score(u, v))
        NMI = squareform(NMI)  # because is distance, it will fill main diagonal with 1
        NMI = np.eye(N=NMI.shape[0]) + NMI  # main diagonal is 1
        sim = pd.DataFrame(NMI, index=header, columns=header)
        return sim.fillna(0)
    elif how == "cosine":
        cos_sim = pdist(data.values.T, "cosine")
        sim = pd.DataFrame(1 - squareform(cos_sim), index=header, columns=header)
        return sim.fillna(0)
    elif how == "angular":
        cos_sim = 1 - squareform(pdist(data.values.T, "cosine"))
        ang_sim = 1 - (np.arccos(cos_sim) / np.pi)
        sim = pd.DataFrame(ang_sim, index=header, columns=header)
        return sim.fillna(0)
    else:
        logger.warning('Choose either "nmi" or "cosine" or "angular".')
        return pd.DataFrame()


def pivot_mps_by_vote(_data, how="outer"):
    """
    :param: how, str, "inner" or "outer". In case of outer, fill nans with 0
    :return a pandas DataFrame having mps as index and votes as columns
    """
    data = _data.copy()
    data["vote"] = data["vote"].replace(to_replace={"y": 1, "n": -1, "o": 0})

    net = pd.DataFrame(
        {"mps": sorted(list(set(data["icpsr"].astype(int).values)))}
    ).set_index("mps")

    for i in data["rollnumber"].unique():
        tmp = (
            data[data["rollnumber"] == i][["icpsr", "vote"]]
            .drop_duplicates()
            .rename(columns={"vote": f"{i}"})
        )
        if how == "inner":
            net = net.join(tmp.set_index("icpsr"), how="inner")
        elif how == "outer":
            net = net.join(tmp.set_index("icpsr"), how="outer").fillna(0)
        else:
            net = pd.DataFrame()

    return net


def get_opinion_groups(year: int):
    """
    :param: year, int (e.g. 2019)
    :return: None
    """
    logger.info(f"PERIOD {period}")

    for chamber in ["Senate", "House"]:
        df = get_individual_votes_by_chamber_and_year(
            senate_or_house=chamber, year=year
        )

        mps = sorted(df["icpsr"].astype(int).unique())
        _topics = sorted(df["policy"].unique())

        clus_layers = pd.DataFrame(columns=mps)

        for topic in _topics:
            pivot = pivot_mps_by_vote(df[df["policy"] == topic].copy(), how="outer")

            # DEFAULT
            mps_cos_sim = get_similarity(pivot, between="mps", how="cosine")

            X = (1 - mps_cos_sim).values
            db = DBSCAN(eps=0.3, min_samples=8, metric="precomputed", n_jobs=-1).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of core samples: %d" % len(db.core_sample_indices_))
            print("Estimated number of noise points: %d" % n_noise_)
            if n_clusters_ > 1:
                print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
            else:
                print("\t one clus: ", set(labels))

            # change -1 labels to unique labels (no -2 labels from here)
            labels = [la if la != -1 else -1 * (i + 2) for i, la in enumerate(labels)]
            # print(labels)
            _mps = [int(m) for m in list(mps_cos_sim.columns)]
            clus_layers = clus_layers.append(dict(zip(_mps, labels)), ignore_index=True)

        labels_df = clus_layers.fillna(-1).T
        labels_df.rename(
            columns=dict(zip(list(labels_df.columns), _topics)), inplace=True
        )

        labels_df.to_csv(
            f"usa_{chamber}_dbscan_{year}.csv",
            sep="\t",
        )
