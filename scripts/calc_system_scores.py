#!/usr/bin/env python3

"""
Usage: ./calc_system_scores.py ../eval/results/parbleu-batch-1-Batch_780_results.csv  ../eval/results/parbleu-batch-2-Batch_781_results.csv ../eval/results/parbleu-batch-3-Batch_782_results.csv  ../eval/results/parbleu-batch-4-Batch_783_results.csv  ../eval/results/parbleu-batch-5-Batch_784_results.csv 
"""

import os, sys
import glob
import pandas as pd
import numpy as np
import argparse


def add_scores(x):
    # Add a score from answer as a separate colum
    _tmp = x["Answer.Q1"].split("|")
    for i in _tmp:
        idx, val = map(int, i.split("__"))
        # Fill zeros as nan-s (missing values)
        if val == 0:
            val = np.nan
        x[f"score_{idx}"] = val
    return x

def normalize_row(row, cols, norm_dict):
    # Normalize the scores with gives mu/std values and create a new column
    for col in cols:
        _tmp = norm_dict[row["Turkle.Username"]]
        mu, std = _tmp["mu"], _tmp["std"]
        new_col = f"norm_{col}"
        row[new_col] = (row[col] - mu) / std
    return row

def main(files):
    dfs = []
    # Read input
    for f in files:
        _parts = f.split("_")
        __parts = _parts[0].split("-")
        _name = f"batch-{__parts[-2]}-{_parts[-2]}"
        df = pd.read_csv(f)
        
        # Extract scores from single string and creates new columns for each row
        df = df.apply(lambda row: add_scores(row), axis=1)
        df["name_id"] = _name
        print(f"Reading input: {_name}, shape={df.shape}.")
    
        dfs.append(df)
    
    # Merge into one big dataframe, clear memory
    df = pd.concat(dfs, join='outer', axis=0)
    del dfs
    
    users = df["Turkle.Username"].unique()
    
    norm_dict = {}
    score_columns = [f"score_{i}" for i in range(100)]
    # find normalization values (mu/std) for each user
    for user in users:
        _values = df[df["Turkle.Username"] == user][score_columns].values
        _mu = np.nanmean(_values)
        _std = np.nanstd(_values)
        norm_dict[user] = {"mu": _mu, "std": _std}
    
    # normalize data and add new columns for data
    df = df.apply(lambda row: normalize_row(row, score_columns, norm_dict), axis=1)
    
    all_models = set()
    
    # Extract models (without which nr of paraphrase was it)
    for i in df[[f"Input.sysname_{i}" for i in range(100)]].values:
        all_models.update(set(i))
    all_models = set("-".join(i.split("-")[:-1]) for i in all_models)
    
    unnorm_results = {mdl_name: [] for mdl_name in all_models}
    norm_results = {mdl_name: [] for mdl_name in all_models}
    
    # For each column, find the subdata that are from a certain model, then add values into a list for easier calculations
    for colN in range(100):
        for model_name in all_models:
            col_system, col_score, norm_col_score = f"Input.sysname_{colN}", f"score_{colN}", f"norm_score_{colN}"
            
            scores = list(df.loc[df[col_system].str.contains(model_name)][col_score].values)
            unnorm_results[model_name] += scores
            scores = list(df.loc[df[col_system].str.contains(model_name)][norm_col_score].values)
            norm_results[model_name] += scores

    # Calculate and print scores
    print()
    print(" ### Unnormalized scores ### ")
    for k,v in unnorm_results.items():
        print(k, np.round(np.nanmean(v), 3))
    print()
    print(" ### Normalized scores ### ")
    for k,v in norm_results.items():
        print(k, np.round(np.nanmean(v), 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate human evaluation average system scores..')
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                    help='Files to evaluate the metrics on')
    args = parser.parse_args()
    main(args.files)
