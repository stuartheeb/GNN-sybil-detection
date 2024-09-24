import copy
import pandas as pd
import os

algorithm_replacements = {
    "SybilRank": "\\textsc{SybilRank}",
    "SybilBelief": "\\textsc{SybilBelief}",
    "SybilSCAR": "\\textsc{SybilSCAR}",
    "SybilGCN_L2_noDropout": "\\textsc{SybilGCN-L2}",
    "SybilGCN-L2": "\\textsc{SybilGCN-L2}",
    "SybilGCN_L4_noDropout": "\\textsc{SybilGCN-L4}",
    "SybilGCN-L4": "\\textsc{SybilGCN-L4}",
    "SybilGCN_L8_noDropout": "\\textsc{SybilGCN-L8}",
    "SybilGCN-L8": "\\textsc{SybilGCN-L8}",
    "SybilRGCN_L2_noDropout": "\\textsc{SybilRGCN-L2}",
    "SybilRGCN-L2": "\\textsc{SybilRGCN-L2}",
    "SybilGAT_L2": "\\textsc{SybilGAT-L2}",
    "SybilGAT-L2": "\\textsc{SybilGAT-L2}",
    "SybilGAT_L4": "\\textsc{SybilGAT-L4}",
    "SybilGAT-L4": "\\textsc{SybilGAT-L4}",
    "SybilGAT_L8": "\\textsc{SybilGAT-L8}",
    "SybilGAT-L8": "\\textsc{SybilGAT-L8}"
}

attack_replacements = {
    "TargetedAttack(p_t=0.05, pdf=[0.25, 0.25, 0.5])": "$p_T=0.05$, $\\boldsymbol p=[0.25,0.25,0.5]$",
    "TargetedAttack(p_t=0.1, pdf=[0.25, 0.25, 0.5])": "$p_T=0.10$, $\\boldsymbol p=[0.25,0.25,0.5]$",
    "TargetedAttack(p_t=0.15, pdf=[0.25, 0.25, 0.5])": "$p_T=0.15$, $\\boldsymbol p=[0.25,0.25,0.5]$",
    "TargetedAttack(p_t=0.2, pdf=[0.25, 0.25, 0.5])": "$p_T=0.20$, $\\boldsymbol p=[0.25,0.25,0.5]$",
}
attack_replacements_presentation = {
    "TargetedAttack(p_t=0.05, pdf=[0.25, 0.25, 0.5])": "$p_t=0.05$, $\\boldsymbol p=[0.25,0.25,0.5]$",
    "TargetedAttack(p_t=0.1, pdf=[0.25, 0.25, 0.5])": "$p_t=0.10$, $\\boldsymbol p=[0.25,0.25,0.5]$",
    "TargetedAttack(p_t=0.15, pdf=[0.25, 0.25, 0.5])": "$p_t=0.15$, $\\boldsymbol p=[0.25,0.25,0.5]$",
    "TargetedAttack(p_t=0.2, pdf=[0.25, 0.25, 0.5])": "$p_t=0.20$, $\\boldsymbol p=[0.25,0.25,0.5]$",
}


def write_experiment_csv(df: pd.DataFrame, directory: str, file_name: str = "data.csv"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(f'{directory}/{file_name}')


def read_experiment_csv(directory: str,
                        file_name: str = "data.csv",
                        make_algorithm_replacements: bool = True,
                        make_attack_replacements: bool = False,
                        presentation_version : bool = False):
    df = pd.read_csv(f'{directory}/{file_name}')
    if make_algorithm_replacements:
        df["algorithm"] = df["algorithm"].replace(algorithm_replacements)

    if make_attack_replacements:
        if presentation_version:
            df["attack"] = df["attack"].replace(attack_replacements_presentation)
        else:
            df["attack"] = df["attack"].replace(attack_replacements)
    return df


def combine_and_copy_lists(list1, list2):
    list = []
    for i in list1:
        list.append(copy.deepcopy(i))
    for i in list2:
        list.append(copy.deepcopy(i))
    return list


def col_title_formatter(title):
    return f"{title}_formatted"


def reformat_dataframe(df):
    df["pretrain_runtime_mean"] = df["pretrain_runtime_mean"] / 1000
    df["pretrain_runtime_std"] = df["pretrain_runtime_std"] / 1000
    df["runtime_mean"] = df["runtime_mean"] / 1000
    df["runtime_std"] = df["runtime_std"] / 1000

    df["runtime_mean"].fillna(0)
    df["runtime_std"].fillna(0)

    df["AUC"] = df["AUC_mean"].map("{:.3f}".format) + " ± " + df["AUC_std"].map("{:.3f}".format)
    df["accuracy"] = df["accuracy_mean"].map("{:.3f}".format) + " ± " + df["accuracy_std"].map("{:.3f}".format)
    df["precision"] = df["precision_mean"].map("{:.3f}".format) + " ± " + df["precision_std"].map("{:.3f}".format)
    df["recall"] = df["recall_mean"].map("{:.3f}".format) + " ± " + df["recall_std"].map("{:.3f}".format)

    df["pretrain_runtime"] = df["pretrain_runtime_mean"].map("{:.3f}".format) + " ± " + df[
        "pretrain_runtime_std"].map("{:.3f}".format)
    df["runtime"] = df["runtime_mean"].map("{:.3f}".format) + " ± " + df["runtime_std"].map("{:.3f}".format)

    return df
