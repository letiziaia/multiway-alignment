import pandas as pd


def load_data(year: int) -> pd.DataFrame:
    """
    Load the data from the given filepath. The file is assumed to be
    a csv file with the columns separated by a semicolon, as in Zenodo.
    See:
        https://zenodo.org/records/12593833
    Cite as:
        Iannucci, L., Faqeeh, A., Salloum, A., Chen, T. H. Y., & Kivelä, M. (2024).
        Multiway Alignment of Twitter networks from 2019 and 2023 Finnish Parliamentary Elections [Data set].
        Zenodo. https://doi.org/10.5281/zenodo.12593833
    :param year: int, the year for the csv file ('finnish_twitter_2019.csv' or 'finnish_twitter_2023.csv')
    :return: pd.DataFrame
    """
    return pd.read_csv(f"finnish_twitter_{year}.csv", sep=";")


def get_parties(year: int) -> pd.DataFrame:
    """
    Get the opinion groups for discussions related to parties from the given year
    :param year: int, the year for the csv file ('finnish_twitter_2019.csv' or 'finnish_twitter_2023.csv')
    :return: pd.DataFrame
    """
    columns = [
        "CENTER",
        "FINNS",
        "GREENS",
        "LEFT",
        "NATIONAL",
        "SDP",
    ]
    return load_data(year=year)[columns]


def get_topics(year: int) -> pd.DataFrame:
    """
    Get the opinion groups for discussions related to topics from the given year
    :param year: int, the year for the csv file ('finnish_twitter_2019.csv' or 'finnish_twitter_2023.csv')
    :return: pd.DataFrame
    """
    columns = [
        "CLIMATE",
        "ECONOMIC_POLICY",
        "EDUCATION",
        "IMMIGRATION",
        "SOCIAL_SECURITY",
    ]
    return load_data(year=year)[columns]
