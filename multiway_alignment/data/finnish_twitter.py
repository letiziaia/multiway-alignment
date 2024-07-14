
import pandas as pd
import networkx as nx
import typing


def load_network(filepath: str) -> pd.DataFrame:
    layer_name = filepath.split("RICH_")[-1].split(".")[0]
    all_users = []
    all_labels = []
    G = nx.read_graphml(filepath)
    labels = nx.get_node_attributes(G, "finetuned_cluster")
    users = nx.get_node_attributes(G, "user_id")
    for node in list(G.nodes()):
        all_users.append(users[node])
        all_labels.append(labels[node])
    return pd.DataFrame({"users": all_users, f"{layer_name}": all_labels}).set_index(
        "users"
    )


def get_parties(files_list: typing.List) -> pd.DataFrame:
    files = [
        f
        for f in files_list
        if "parties" not in f.lower()
        and "climate" not in f.lower()
        and "economic" not in f.lower()
        and "education" not in f.lower()
        and "immigration" not in f.lower()
        and "social" not in f.lower()
    ]
    df = pd.DataFrame()
    for f in files:
        if df.empty:
            df = load_network(f)
        else:
            df = pd.merge(
                df, load_network(f), how="outer", left_index=True, right_index=True
            )

    df = df.reset_index(drop=True)
    name = [c.split("\\")[-1] for c in df.columns]
    name = ["_".join(n.split("_")[:-2]) for n in name]
    df.columns = name  # type: ignore
    return df


def get_topics(files_list: typing.List) -> pd.DataFrame:
    files = [
        f
        for f in files_list
        if "climate" in f.lower()
        or "economic" in f.lower()
        or "education" in f.lower()
        or "immigration" in f.lower()
        or "social" in f.lower()
    ]
    df = pd.DataFrame()
    for f in files:
        if df.empty:
            df = load_network(f)
        else:
            df = pd.merge(
                df, load_network(f), how="outer", left_index=True, right_index=True
            )

    df = df.reset_index(drop=True)
    name = [c.split("\\")[-1] for c in df.columns]
    name = ["_".join(n.split("_")[:-2]) for n in name]
    df.columns = name  # type: ignore
    return df


def get_all(files_list: typing.List) -> pd.DataFrame:
    df = pd.DataFrame()
    for f in files_list:
        if df.empty:
            df = load_network(f)
        else:
            df = pd.merge(
                df, load_network(f), how="outer", left_index=True, right_index=True
            )

    df = df.reset_index(drop=True)
    name = [c.split("\\")[-1] for c in df.columns]
    name = ["_".join(n.split("_")[:-2]) for n in name]
    df.columns = name  # type: ignore
    return df
