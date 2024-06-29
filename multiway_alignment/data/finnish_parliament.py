import os
import requests  # type: ignore
import requests.exceptions  # type: ignore
import pandas as pd

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
    hasNext = True
    while hasNext:
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
        hasNext = res.get("hasMore")
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
        hasNext = True
        while hasNext:
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
            hasNext = res.get("hasMore")
            # update page number
            page += 1


if __name__ == "__main__":
    y = 2007
    while y >= 1996:
        get_mps_votes(year=y)
        y -= 1
    get_table_of_votes()
