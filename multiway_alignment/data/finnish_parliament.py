import os
import requests  # type: ignore
import requests.exceptions  # type: ignore
import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from multilayer_alignment.utils.logging import logger  # type: ignore


def all_votes(pagenum):
    return f"/SaliDBAanestys/rows?perPage=100&page={pagenum}"


def mps_opinions(pagenum, voteid):
    return f"/SaliDBAanestysEdustaja/rows?perPage=100&page={pagenum}&columnName=AanestysId&columnValue={voteid}"


def get_table_of_votes():
    """
    Download the data of the votes of the Finnish parliament
    :return: None
    """
    save_into_folder = "eduskunta"
    if not os.path.exists(save_into_folder):
        os.makedirs(save_into_folder)

    base_url = "https://avoindata.eduskunta.fi/api/v1/tables"
    aanestys_expected_columns = [
        "AanestysId",
        "KieliId",
        "IstuntoVPVuosi",
        "IstuntoNumero",
        "IstuntoPvm",
        "IstuntoIlmoitettuAlkuaika",
        "IstuntoAlkuaika",
        "PJOtsikko",
        "AanestysNumero",
        "AanestysAlkuaika",
        "AanestysLoppuaika",
        "AanestysMitatoity",
        "AanestysOtsikko",
        "AanestysLisaOtsikko",
        "PaaKohtaTunniste",
        "PaaKohtaOtsikko",
        "PaaKohtaHuomautus",
        "KohtaKasittelyOtsikko",
        "KohtaKasittelyVaihe",
        "KohtaJarjestys",
        "KohtaTunniste",
        "KohtaOtsikko",
        "KohtaHuomautus",
        "AanestysTulosJaa",
        "AanestysTulosEi",
        "AanestysTulosTyhjia",
        "AanestysTulosPoissa",
        "AanestysTulosYhteensa",
        "Url",
        "AanestysPoytakirja",
        "AanestysPoytakirjaUrl",
        "AanestysValtiopaivaasia",
        "AanestysValtiopaivaasiaUrl",
        "AliKohtaTunniste",
        "Imported",
    ]

    # init file
    file_name = save_into_folder + "/votes_by_year.csv"
    pd.DataFrame(columns=aanestys_expected_columns).to_csv(file_name, index=False)

    page = 0
    has_next = True
    while has_next:
        logger.info(f"Downloading page {page}")
        first_response = requests.get(base_url + all_votes(f"{page}"))
        res = first_response.json()
        # check validity
        assert res.get("columnNames", []) == aanestys_expected_columns
        # get data
        response_data = res.get("rowData")
        data = []
        for row in response_data:
            # data.append({k: row[v] for k, v in col_names_to_idx.items()})
            data.append({c: row[i] for i, c in enumerate(aanestys_expected_columns)})
        pd.DataFrame(data).to_csv(file_name, mode="a", header=False, index=False)
        # update flag
        has_next = res.get("hasMore")
        # update page number
        page += 1


def get_mps_votes(year: int = 2019) -> None:
    """
    Download the data of the votes of the Finnish parliament for a given year
    :param year: int, the year to download the data for
        Default: 2019
    :return: None
    """
    logger.info(f"Downloading votes of the Finnish parliament for the year {year}")
    base_url = "https://avoindata.eduskunta.fi/api/v1/tables"
    mps_votes_expected_columns = [
        "EdustajaId",
        "AanestysId",
        "EdustajaEtunimi",
        "EdustajaSukunimi",
        "EdustajaHenkiloNumero",
        "EdustajaRyhmaLyhenne",
        "EdustajaAanestys",
        "Imported",
    ]

    # init file
    file_name = f"eduskunta/{year}_mps_votes.csv"
    if os.path.isfile(file_name):
        logger.warning("The current file will be replaced.")
    pd.DataFrame(columns=["year"] + mps_votes_expected_columns).to_csv(
        file_name, index=False
    )

    # find query details
    votes_file = "eduskunta/votes_by_year.csv"
    if not os.path.isfile(votes_file):
        get_table_of_votes()

    votes_df = pd.read_csv(votes_file)
    votes_ids = votes_df[votes_df["IstuntoVPVuosi"] == year]["AanestysId"].values

    for i, v in enumerate(votes_ids):
        logger.info(f"Downloading vote {v} ({i+1} of {len(votes_ids)})")
        page = 0
        has_next = True
        while has_next:
            first_response = requests.get(base_url + mps_opinions(f"{page}", f"{v}"))
            res = first_response.json()
            # check validity
            assert res.get("columnNames", []) == mps_votes_expected_columns
            # get data
            response_data = res.get("rowData")
            data = []
            for row in response_data:
                data.append(
                    {"year": year}
                    | {k: row[i] for i, k in enumerate(mps_votes_expected_columns)}
                )
            pd.DataFrame(data).to_csv(file_name, mode="a", header=False, index=False)
            # update flag
            has_next = res.get("hasMore")
            # update page number
            page += 1


topics = {
    "economics": [
        "vero",
        "työ",
        "talous",
        "talouden",
        "yritys",
        "korvau",
        "elinkeino",
        "laina",
    ],
    "health": [
        "tervey",
        "tartunta",
        "korona",
        "covid",
        "alkoholi",
        "silpomis",
        "palvelusetel",
    ],
    "climate": [
        "ilmasto",
        "henkilöauto",
        "sähköajo",
        "neste",
        "poltto",
        "romutuspalkio",
        "energia",
    ],
    "education": ["oppi", "varhaiskasvat", "koulu", "esiopetu", "ammatillis"],
    "immigration": ["kotoutumis", "maahantulo"],
    "unemployment": ["toimeentulo", "työttömyys"],
    "culture": ["kulttuur"],
    "social_issues": ["syrjäytymis", "nuort"],
    # "general_explanatory_statements": ["yleisperustelu"],
    "communications_and_transportation": ["viestin", "tieliikenne"],
    "defense": ["puolustus"],
    "foreign_affairs": ["ulkominister", "ulkomaalais"],
}


first_dates_cutpoints = [
    "1995-03-22",  ## s
    "1996-03-28",
    "1997-04-04",
    "1998-04-11",
    "1999-04-20",  ## s
    "2000-04-11",
    "2001-04-03",
    "2002-03-26",
    "2003-03-18",  ## s
    "2004-03-31",
    "2005-04-14",
    "2006-04-28",
    "2007-03-28",
    # "2007-05-15", ## s
    "2008-05-08",
    "2009-05-02",
    "2010-04-26",
    "2011-04-21",  ## s
    "2012-04-23",
    "2013-04-26",
    "2014-04-29",
    "2015-05-05",  ## s
    "2016-05-04",
    "2017-05-04",
    "2018-05-04",
    "2019-05-07",  ## s
    "2020-05-06",
    "2021-05-06",
]


party_list = ["kesk", "kok", "ps", "r", "sd", "vas", "vihr", "kd"]


def at_least_one(sentence: str, words: list[str]) -> bool:
    """
    Return True if at least one word from the list is in the sentence.
    :param: sentence, str
    :param: words, list of str
    :return: bool
    """
    flag = False
    for w in words:
        if w in sentence.lower():
            flag = True
    return flag


def load_votes(period: str) -> pd.DataFrame:
    """
    :param: period, str (e.g. "2019-2020")
    :return: a pandas DataFrame
    """
    all_votes = pd.read_csv(
        "eduskunta/votes_by_year.csv", sep=",", low_memory=False
    ).dropna(subset=["KohtaOtsikko"])
    all_votes["IstuntoPvm"] = pd.to_datetime(all_votes["IstuntoPvm"])
    # only finnish
    all_votes = all_votes[all_votes["KieliId"] == 1]

    # filter by period
    start_year, end_year = period.split("-")
    cut_at = sorted(
        [d for d in first_dates_cutpoints if d[:4] == start_year or d[:4] == end_year]
    )
    logger.info(f"cutpoints for {period}: {cut_at}")
    if len(cut_at) < 2:
        all_votes = all_votes[
            (all_votes["IstuntoPvm"] >= pd.to_datetime(cut_at[0]))
        ].copy()
    else:
        all_votes = all_votes[
            (all_votes["IstuntoPvm"] >= pd.to_datetime(cut_at[0]))
            & (all_votes["IstuntoPvm"] <= pd.to_datetime(cut_at[1]))
        ].copy()

    all_votes["period"] = [period] * len(all_votes)

    votes_topics = [
        [t for t, kwd in topics.items() if at_least_one(title, kwd)]
        for title in all_votes["KohtaOtsikko"].fillna("unknown").values
    ]
    # fill remaining
    votes_topics = [d if len(d) > 0 else ["other"] for d in votes_topics]
    # remove econ if better match
    votes_topics = [
        (
            " ".join(d).replace("economics", "").split()
            if len(d) > 1 and "economics" in d
            else d
        )
        for d in votes_topics
    ]
    # remove ulko if better match
    votes_topics = [
        (
            " ".join(d).replace("ulkominister", "").split()
            if len(d) > 1 and "ulkominister" in d
            else d
        )
        for d in votes_topics
    ]
    # remove education if better match
    votes_topics = [
        (
            " ".join(d).replace("education", "").split()
            if len(d) > 1 and "education" in d
            else d
        )
        for d in votes_topics
    ]
    # remove list
    votes_topics = [" ".join(d) for d in votes_topics]  # type: ignore
    # ensure one topic only
    votes_topics = [d.split()[0] for d in votes_topics]  # type: ignore
    all_votes["topic"] = votes_topics

    # exclude other, yleis
    all_votes = all_votes[~all_votes["topic"].isin(["other", "yleisperustelu"])].copy()

    return all_votes[
        [
            "AanestysId",
            "IstuntoPvm",
            "IstuntoVPVuosi",
            "period",
            "KohtaOtsikko",
            "topic",
        ]
    ]


def load_elec_period(period: str):
    """
    :param: period, str (e.g. "2019-2020")
    :return: a pandas DataFrame
    """
    votes = load_votes(period)
    mps_votes = []
    for year in period.split("-"):
        try:
            mps_votes.append(pd.read_csv(f"eduskunta/{year}_mps_votes.csv", sep=","))
        except Exception as e:
            logger.error(e)

    period_df = pd.concat(mps_votes, ignore_index=True)
    for c in period_df.columns:
        if c not in ["year", "Imported"]:
            try:
                period_df[c] = period_df[c].str.strip()
            except Exception as e:
                logger.warning(e)

    period_df["year"] = period_df["year"].astype(int)
    period_df["EdustajaHenkiloNumero"] = period_df["EdustajaHenkiloNumero"].astype(int)
    period_df["EdustajaRyhmaLyhenne"] = period_df["EdustajaRyhmaLyhenne"].str.strip()
    period_df["EdustajaRyhmaLyhenne"] = period_df["EdustajaRyhmaLyhenne"].str.lower()

    period_df = period_df[period_df["EdustajaRyhmaLyhenne"].isin(party_list)].copy()

    logger.info(
        f"All MPs (8 biggest parties): {period_df['EdustajaHenkiloNumero'].nunique()}"
    )
    logger.info(f"nans {period_df.isnull().sum().sum()}")
    logger.info(f"# unique votes {period_df['AanestysId'].nunique()}")

    period_df["EdustajaAanestys"].replace(
        {
            "Poissa": 0,
            "Jaa": 1,
            "Ei": -1,
            "Tyhjää": 0,
            "Blank": np.nan,
            "Ja": np.nan,
            "Nej": np.nan,
            "Frånvarande": np.nan,
            "Avstår": np.nan,
        },
        inplace=True,
    )
    period_df.dropna(inplace=True)
    period_df["EdustajaAanestys"] = period_df["EdustajaAanestys"].astype(int)

    period_df = pd.merge(
        left=period_df,
        right=votes,
        how="inner",
        left_on=["year", "AanestysId"],
        right_on=["IstuntoVPVuosi", "AanestysId"],
    )

    return period_df[~period_df["topic"].isin(["other", "yleisperustelu"])].copy()


def get_similarity(data: pd.DataFrame, between: str, how: str) -> pd.DataFrame:
    """
    :param: data, a pandas DataFrame for one election period
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
        return pd.DataFrame()

    header = list(data.columns)
    if how == "nmi":
        NMI = pdist(data.values.T, lambda u, v: normalized_mutual_info_score(u, v))
        NMI = squareform(NMI)  # because is distance, it will fill main diagonal with 0
        NMI = (
            np.eye(N=NMI.shape[0]) + NMI
        )  # but we want similarity, so main diagonal is 1
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


def pivot_mps_by_vote(data, how="outer"):
    """
    :param: data, a pandas DataFrame for one election period
    :param: how, str, "inner" or "outer". In case of outer, fill nans with 0
    :return a pandas DataFrame having mps as index and votes as columns
    """
    logger.info(f"mps: {data['EdustajaHenkiloNumero'].nunique()}")
    logger.info(f"votes: {data['AanestysId'].nunique()}")
    net = pd.DataFrame(
        {"mps": sorted(list(set(data["EdustajaHenkiloNumero"].astype(int).values)))}
    ).set_index("mps")

    for i in data["AanestysId"].unique():
        tmp = (
            data[data["AanestysId"] == i][["EdustajaHenkiloNumero", "EdustajaAanestys"]]
            .drop_duplicates()
            .rename(columns={"EdustajaAanestys": f"{i}"})
        )
        if how == "inner":
            net = net.join(tmp.set_index("EdustajaHenkiloNumero"), how="inner")
        elif how == "outer":
            net = net.join(tmp.set_index("EdustajaHenkiloNumero"), how="outer").fillna(
                0
            )
        else:
            net = pd.DataFrame()

    return net


def get_opinion_groups(period: str):
    """
    :param: period, str
    :return: None
    """
    logger.info(f"PERIOD {period}")

    df = load_elec_period(period=period)

    mps = sorted(df["EdustajaHenkiloNumero"].astype(int).unique())
    _topics = sorted(df["topic"].unique())

    logger.info(f"n_topics: {len(_topics)}")

    clus_layers = pd.DataFrame(columns=mps)

    for topic in _topics:
        pivot = pivot_mps_by_vote(df[df["topic"] == topic].copy(), how="outer")

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
        _mps = [int(m) for m in list(mps_cos_sim.columns)]
        clus_layers = clus_layers.append(dict(zip(_mps, labels)), ignore_index=True)  # type: ignore

    labels_df = clus_layers.fillna(-1).T
    labels_df.rename(columns=dict(zip(list(labels_df.columns), _topics)), inplace=True)

    labels_df.to_csv(
        f"eduskunta_dbscan_{period}.csv",
        sep="\t",
    )


if __name__ == "__main__":
    y = 2007
    while y >= 1996:
        get_mps_votes(year=y)
        y -= 1
    get_table_of_votes()
