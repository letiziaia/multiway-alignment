import pandas as pd
import numpy as np

import multiway_alignment.score as mw_score
import multiway_alignment.null_models as mw_null
from joblib import dump  # type: ignore


def get_anes_data() -> pd.DataFrame:
    """
    Load the ANES data
    :return: pd.DataFrame
    """
    path = "../../all_survey_data/anes_timeseries_cdf_csv_20220916.csv"

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df[
        [
            "Version",
            "VCF0004",
            "VCF0006",
            "VCF9223",
            "VCF0838",
            "VCF9239",
            "VCF9229",
            "VCF9222",
            "VCF9205",
        ]
    ].replace(" ", np.nan).dropna(inplace=True)

    q = {
        "VCF9223": "immigration",  # 1 2 3 4
        "VCF0838": "abortion",  # 1 2 3 4
        "VCF9238": "gun_access",  # 1 2 3 4 5
        "VCF9229": "unemployment",  # 1 2 3
        "VCF9205": "which_party",  # 1 2 (3, 7)
        "VCF9236": "death_penalty",  # 1 2 (-8, -9)
        "VCF0876": "homosexual_discrim",  # 1 2
        "VCF0839": "government_spending",  # 1 to 7
    }

    timeseries = (
        df[["Version", "VCF0004", "VCF0006"] + list(q.keys())]
        .replace(" ", np.nan)
        .dropna()
        .rename(columns=q)
        .copy()
    )

    for c in q.values():
        timeseries[c] = (
            timeseries[c]
            .astype(str)
            .str.replace(" ", "")
            .replace(to_replace="", value=np.nan)
            .astype(float)
        )
    timeseries.dropna(axis=0, inplace=True)

    timeseries["abortion"] = timeseries["abortion"].replace(9, 0)
    timeseries["homosexual_discrim"] = timeseries["homosexual_discrim"].replace(9, 8)
    timeseries["which_party"] = timeseries["which_party"].replace(7, 3)
    timeseries["government_spending"] = (
        timeseries["government_spending"]
        .replace(2, 1)
        .replace(3, 1)
        .replace(5, 7)
        .replace(6, 7)
        .replace(9, 0)
    )
    for c in q.values():
        timeseries[c] = timeseries[c].clip(lower=0)

    timeseries = timeseries.reset_index(drop=True)

    return timeseries


def compute_all_alignments(timeseries: pd.DataFrame) -> None:
    """
    Compute all alignments for the ANES data
    :param timeseries: pd.DataFrame, the ANES data
    :return: None
    """
    for year in sorted(timeseries["VCF0004"].unique()):
        print(f"\n\n--> YEAR {year}")
        _df = (
            timeseries[timeseries["VCF0004"] == year][sorted(timeseries.columns[3:])]
            .copy()
            .reset_index(drop=True)
        )
        dump_name = f"survey_{year}"
        full, _ = mw_score.maximal_alignment_curve_nminusone(
            opinions=_df,
            which_score="ami",
            adjusted=False,
            dump_to=dump_name + "_nminus1_ami",
        )
        dump(full, dump_name + "_nminus1_ami_full")

        print("null model")
        mw_null.random_full_alignment_curves_kminusone(
            df=_df,
            save_to=dump_name + "_nminus1_null",
            which_score="ami",
            adjusted=False,
            n_tries=10,
        )


if __name__ == "__main__":
    data = get_anes_data()
    compute_all_alignments(data)
