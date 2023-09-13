#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from itertools import combinations

def printf(*args):
    print(*args)

def fn_ctrl_rename(x):
    return f"ctrl_{x}"

def fn_diff_rename(x):
    return f"diff_{x}"

def fn_diff_i_rename(x, i):
    return f"diff{i}_{x}"

def fn_lag_rename(x, k):
    return f"L{k}_{x}"

def fn_lead_rename(x, k):
    return f"F{k}_{x}"

def get_controls_rename(controls):
    return [fn_ctrl_rename(x) for x in controls]

def get_diff_controls_rename(controls):
    return [fn_diff_rename(fn_ctrl_rename(x)) for x in controls]

def generate_return_names(dynamic=0, placebo=0):
    names = []

    if placebo > 0:
        for i in range(placebo, 0, -1):
            nm_effect_placebo = f"placebo_{i}"
            names.append(nm_effect_placebo)

    names.append("effect")

    if dynamic > 0:
        for i in range(1, dynamic + 1):
            nm_effect_dynamic = f"dynamic_{i}"
            names.append(nm_effect_dynamic)

    return names

def generate_switchers_names(dynamic=0, placebo=0):
    names = []

    if placebo > 0:
        for i in range(placebo, 0, -1):
            names.append(None)

    nm_N_switchers = "N_switchers_effect"
    names.append(nm_N_switchers)

    if dynamic > 0:
        for i in range(1, dynamic + 1):
            nm_N_switchers_dynamic = f"N_switchers_effect_dynamic_{i}"
            names.append(nm_N_switchers_dynamic)

    return names

def did_multiplegt_rename_var(df, Y, G, T, D, controls, recat_treatment, trends):
    controls_rename = get_controls_rename(controls)
    original_names = [Y, G, T, D] + controls
    new_names = ["Y", "G", "T", "D"] + controls_rename
    df = df.rename(columns=dict(zip(original_names, new_names)))

    if recat_treatment is not None:
        df["Drecat"] = df[recat_treatment]

    if trends is not None:
        df["Vtrends"] = df[trends]

    df = df[new_names]
    return df

def did_multiplegt_transform(df, controls, trends_nonparam):
    df_grouped = df.groupby(["G", "T"])
    df["counter"] = df_grouped["Y"].transform("count")
    tdf = df.groupby("T")["T"].transform(lambda x: pd.factorize(x)[0] + 1)
    df["T"] = tdf

    ignore_columns = ["G", "T", "tag"]
    if "Vtrends" in df.columns:
        ignore_columns.append("Vtrends")

    df["tag"] = df[controls].applymap(np.isnan).sum(axis=1)
    df[ignore_columns] = df[ignore_columns].applymap(lambda x: np.nan if x else x)
    cat_treatment = "Drecat" if "Drecat" in df.columns else "D"
    tdf = df.groupby(cat_treatment)[cat_treatment].transform(lambda x: pd.factorize(x)[0] + 1)
    df["DgroupLag"] = tdf

    columns_to_diff = ["Y", "D"] + get_controls_rename(controls)
    df_diff = df.groupby("G")[columns_to_diff].transform(lambda x: x - x.shift(1))
    df_diff = df_diff.rename(columns=dict(zip(columns_to_diff, get_diff_controls_rename(columns_to_diff))))
    df = pd.concat([df, df_diff], axis=1)

    df["Tfactor"] = df["T"]

    if "Vtrends" in df.columns:
        df["Vtrends"] = df["Vtrends"].astype("category")

    if trends_nonparam:
        tdf = df.groupby(["Vtrends", "Tfactor"]).size().reset_index(name="VtrendsX")
        df = df.merge(tdf, on=["Vtrends", "Tfactor"])
        df["Vtrends"] = df["VtrendsX"]
        df = df.drop(columns=["VtrendsX"])

    return df

def placebo_transform_level(df, counter_placebo, controls):
    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    df_l = df.groupby("G")[lag_names].shift(counter_placebo)
    df_l = df_l.rename(columns=dict(zip(lag_names, [fn_lag_rename(x, counter_placebo) for x in lag_names])))
    df = pd.concat([df, df_l], axis=1)
    return df

def dynamic_transform_level(df, counter_dynamic, controls):
    nm_switchers = generate_switchers_names(counter_dynamic, 0)
    df = pd.concat([df, pd.DataFrame(columns=nm_switchers)], axis=1)
    df["N_switchers_effect"] = 1
    for i in range(1, counter_dynamic + 1):
        df[f"N_switchers_effect_dynamic_{i}"] = 0

    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    df_l = df.groupby("G")[lag_names].shift(counter_dynamic + 1)
    df_l = df_l.rename(columns=dict(zip(lag_names, [fn_lead_rename(x, counter_dynamic) for x in lag_names])))
    df = pd.concat([df, df_l], axis=1)
    return df

def dynamic_transform_level_placebo(df, counter_placebo, counter_dynamic, controls):
    nm_switchers = generate_switchers_names(counter_dynamic, counter_placebo)
    df = pd.concat([df, pd.DataFrame(columns=nm_switchers)], axis=1)
    df["N_switchers_effect"] = 1
    for i in range(1, counter_dynamic + 1):
        df[f"N_switchers_effect_dynamic_{i}"] = 0

    for i in range(1, counter_placebo + 1):
        df[f"placebo_{i}"] = 0

    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    df_l = df.groupby("G")[lag_names].shift(counter_dynamic + 1)
    df_l = df_l.rename(columns=dict(zip(lag_names, [fn_lead_rename(x, counter_dynamic) for x in lag_names])))
    df = pd.concat([df, df_l], axis=1)

    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    df_l = df.groupby("G")[lag_names].shift(counter_placebo)
    df_l = df_l.rename(columns=dict(zip(lag_names, [fn_lag_rename(x, counter_placebo) for x in lag_names])))
    df = pd.concat([df, df_l], axis=1)
    return df

def dynamic_transform_level_placebo_trends(df, counter_placebo, counter_dynamic, controls):
    nm_switchers = generate_switchers_names(counter_dynamic, counter_placebo)
    df = pd.concat([df, pd.DataFrame(columns=nm_switchers)], axis=1)
    df["N_switchers_effect"] = 1
    for i in range(1, counter_dynamic + 1):
        df[f"N_switchers_effect_dynamic_{i}"] = 0

    for i in range(1, counter_placebo + 1):
        df[f"placebo_{i}"] = 0

    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    df_l = df.groupby("G")[lag_names].shift(counter_dynamic + 1)
    df_l = df_l.rename(columns=dict(zip(lag_names, [fn_lead_rename(x, counter_dynamic) for x in lag_names])))
    df = pd.concat([df, df_l], axis=1)

    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    df_l = df.groupby("G")[lag_names].shift(counter_placebo)
    df_l = df_l.rename(columns=dict(zip(lag_names, [fn_lag_rename(x, counter_placebo) for x in lag_names])))
    df = pd.concat([df, df_l], axis=1)

    df_trends = df.groupby(["G", "T"])["Vtrends"].transform("count")
    tdf = df.groupby("G")["Vtrends"].transform(lambda x: pd.factorize(x)[0] + 1)
    df["Vtrends"] = tdf
    return df

def build_plug_vector(df, counter_dynamic, counter_placebo):
    nm_switchers = generate_switchers_names(counter_dynamic, counter_placebo)
    pvec = np.zeros((counter_dynamic + counter_placebo + 1, len(nm_switchers)))
    pvec[0, 0] = 1
    for i in range(1, counter_dynamic + 1):
        pvec[i, i] = 1

    for i in range(1, counter_placebo + 1):
        pvec[counter_dynamic + i, counter_dynamic] = 1

    return pvec

def build_vector_diff_lag(df, counter_dynamic, counter_placebo, controls):
    lag_names = ["diff_D", "diff_Y"] + get_diff_controls_rename(controls)
    pvec = np.zeros((counter_dynamic + counter_placebo + 1, len(lag_names)))
    pvec[0, :] = 1

    for i in range(1, counter_dynamic + 1):
        pvec[i, :] = 1

    for i in range(1, counter_placebo + 1):
        pvec[counter_dynamic + i, :] = -1

    return pvec

def select_formula(nm_switchers, counter_dynamic, counter_placebo):
    formula = "N_switchers_effect ~ 1"
    for i in range(1, counter_dynamic + counter_placebo + 1):
        formula += f" + {nm_switchers[i]}"
    return formula

def selection_matrix(nm_switchers, counter_dynamic, counter_placebo, trends):
    pvec = np.zeros((counter_dynamic + counter_placebo + 1, len(nm_switchers)))
    pvec[0, 0] = 1
    for i in range(1, counter_dynamic + 1):
        pvec[i, i] = 1

    for i in range(1, counter_placebo + 1):
        pvec[counter_dynamic + i, counter_dynamic] = 1

    if trends is not None:
        pvec[0, 0] = 0
        for i in range(1, counter_dynamic + counter_placebo + 1):
            pvec[i, i] = 0
    return pvec

def prepare_df(df, controls, placebo, dynamic, trends, trends_nonparam):
    if placebo == 0 and dynamic == 0 and trends is None:
        df = did_multiplegt_rename_var(df, "Y", "G", "T", "D", controls, None, None)
        df = did_multiplegt_transform(df, controls, trends_nonparam)
    elif placebo > 0 and dynamic == 0 and trends is None:
        df = did_multiplegt_rename_var(df, "Y", "G", "T", "D", controls, "D", None)
        df = did_multiplegt_transform(df, controls, trends_nonparam)
        df = placebo_transform_level(df, placebo, controls)
    elif placebo == 0 and dynamic > 0 and trends is None:
        df = did_multiplegt_rename_var(df, "Y", "G", "T", "D", controls, None, None)
        df = did_multiplegt_transform(df, controls, trends_nonparam)
        df = dynamic_transform_level(df, dynamic, controls)
    elif placebo > 0 and dynamic > 0 and trends is None:
        df = did_multiplegt_rename_var(df, "Y", "G", "T", "D", controls, "D", None)
        df = did_multiplegt_transform(df, controls, trends_nonparam)
        df = dynamic_transform_level_placebo(df, placebo, dynamic, controls)
    elif placebo == 0 and dynamic == 0 and trends is not None:
        df = did_multiplegt_rename_var(df, "Y", "G", "T", "D", controls, None, None)
        df = did_multiplegt_transform(df, controls, trends_nonparam)
        df = dynamic_transform_level(df, dynamic, controls)
        df = dynamic_transform_level_placebo_trends(df, placebo, dynamic, controls)
    else:
        raise ValueError("Invalid combination of controls, placebo, dynamic, and trends.")
    return df

def run_did(df, controls, placebo, dynamic, trends, trends_nonparam):
    df = prepare_df(df, controls, placebo, dynamic, trends, trends_nonparam)
    formula = select_formula(generate_switchers_names(dynamic, placebo), dynamic, placebo)
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    W = selection_matrix(generate_switchers_names(dynamic, placebo), dynamic, placebo, trends)
    result = IV2SLS(y, X, None, None, W).fit(cov_type="kernel", kernel="bartlett")
    return result

# Example usage:
result = run_did(df, controls=["X1", "X2"], placebo=1, dynamic=1, trends=["T1", "T2"], trends_nonparam=["T3", "T4"])
print(result.summary())

