#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_c_rc_input_and_label.py
------------------------------
• Load c-rc and ZF tables (`c_rc_df.csv`, `zf_data_df.csv`)
• Build neighbour windows:
      20 aa  = 4 left  + 12 core + 4 right
      26 aa  = 7 left  + 12 core + 7 right
      36 aa  = 12 left + 12 core + 12 right
  (padding with “X” whenever the adjacent finger is missing)
• One-hot-encode every window (20-long vector per residue, “X” = 0.05)
• Save .npy/.csv files under ../../Data/PWMpredictor/new_data/
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
from functions import *          # oneHot_Amino_acid_vec, etc.

# ----------------------------------------------------------------------
# Load raw tables (space-delimited)
# ----------------------------------------------------------------------
c_rc_df   = pd.read_csv("../../Data/PWMpredictor/c_rc_df.csv",
                        sep=r"\s+", engine="python")
zf_data_df = pd.read_csv("../../Data/PWMpredictor/zf_data_df.csv",
                         sep=r"\s+", engine="python")

# Column names
CRC_PROT_COL, CRC_IDX_COL = "UniProt_ID", "ZF_index"
ZF_PROT_COL,  ZF_IDX_COL  = "prot_name_id", "zf_index"

# ----------------------------------------------------------------------
# Neighbour-window helper
# ----------------------------------------------------------------------
def add_neighbour_window(df: pd.DataFrame,
                         prot_col: str,
                         idx_col: str,
                         k_left: int,
                         k_right: int,
                         pad_char: str = "X") -> pd.Series:
    """
    Build a neighbour window of size  k_left + 12 + k_right.
    If the previous/next finger is not contiguous, use pad_char.
    """
    pad_left  = pad_char * k_left
    pad_right = pad_char * k_right

    df = df.sort_values([prot_col, idx_col]).reset_index(drop=True)

    prev_seq = df.groupby(prot_col)["res_12"].shift(1)
    prev_idx = df.groupby(prot_col)[idx_col].shift(1)
    next_seq = df.groupby(prot_col)["res_12"].shift(-1)
    next_idx = df.groupby(prot_col)[idx_col].shift(-1)

    out = []
    for row, p_seq, p_idx, n_seq, n_idx in zip(
            df.itertuples(index=False), prev_seq, prev_idx, next_seq, next_idx):

        prev_ok = pd.notna(p_seq) and (row._asdict()[idx_col] - 1 == p_idx)
        next_ok = pd.notna(n_seq) and (row._asdict()[idx_col] + 1 == n_idx)

        left  = p_seq[-k_left:] if (k_left and prev_ok) else pad_left
        right = n_seq[:k_right] if (k_right and next_ok) else pad_right
        out.append(f"{left}{row.res_12}{right}")

    return pd.Series(out, index=df.index)

# ----------------------------------------------------------------------
# Create neighbour windows for the ZF table
# ----------------------------------------------------------------------
for length, flanks in {"20": (4, 4), "26": (7, 7), "36": (12, 12)}.items():
    l, r = flanks
    zf_data_df[f"res_{length}_neighbors"] = add_neighbour_window(
        zf_data_df, ZF_PROT_COL, ZF_IDX_COL, l, r
    )

# ----------------------------------------------------------------------
# Copy neighbour windows into the c-rc table
# ----------------------------------------------------------------------
merge_cols = [ZF_PROT_COL, ZF_IDX_COL]
c_rc_df = c_rc_df.merge(
    zf_data_df[merge_cols + [f"res_{n}_neighbors" for n in ("20", "26", "36")]],
    left_on=[CRC_PROT_COL, CRC_IDX_COL],
    right_on=merge_cols,
    how="left"
).drop(columns=merge_cols)

# Guarantee every row has a full-length window (pad with X if missing)
def pad_window(seq: str, total_len: int) -> str:
    need = total_len - len(seq)
    return seq + ("X" * need)

for length, total in {"20": 20, "26": 26, "36": 36}.items():
    col = f"res_{length}_neighbors"
    c_rc_df[col].fillna(
        c_rc_df["res_12"].apply(lambda core, t=total: pad_window("X"* ((t-12)//2) + core, t)),
        inplace=True
    )

# ----------------------------------------------------------------------
# One-hot encodings   (c-rc  +  ZF)
# ----------------------------------------------------------------------
one_hot_c_rc = {
    "4"       : oneHot_Amino_acid_vec(c_rc_df["res_4"]),
    "7"       : oneHot_Amino_acid_vec(c_rc_df["res_7"]),
    "7b1h"    : oneHot_Amino_acid_vec(c_rc_df["res_7_b1h"]),
    "12"      : oneHot_Amino_acid_vec(c_rc_df["res_12"]),
    "20neigh" : oneHot_Amino_acid_vec(c_rc_df["res_20_neighbors"]),
    "26neigh" : oneHot_Amino_acid_vec(c_rc_df["res_26_neighbors"]),
    "36neigh" : oneHot_Amino_acid_vec(c_rc_df["res_36_neighbors"]),
}

one_hot_zf = {
    "20neigh" : oneHot_Amino_acid_vec(zf_data_df["res_20_neighbors"]),
    "26neigh" : oneHot_Amino_acid_vec(zf_data_df["res_26_neighbors"]),
    "36neigh" : oneHot_Amino_acid_vec(zf_data_df["res_36_neighbors"]),
}

# ----------------------------------------------------------------------
# PWM labels (unchanged)
# ----------------------------------------------------------------------
pwm_cols = ['A1','C1','G1','T1','A2','C2','G2','T2','A3','C3','G3','T3']
pwm = c_rc_df[pwm_cols].values

# ----------------------------------------------------------------------
# Save to disk
# ----------------------------------------------------------------------
out_dir = Path("../../Data/PWMpredictor/new_data")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "ground_truth_c_rc.npy", pwm)

for k, arr in one_hot_c_rc.items():
    np.save(out_dir / f"onehot_encoding_c_rc_{k}.npy", arr)

for k, arr in one_hot_zf.items():
    np.save(out_dir / f"onehot_encoding_zf_{k}.npy", arr)

c_rc_df.to_csv(out_dir / "c_rc_df.csv", sep=" ", index=False)
zf_data_df.to_csv(out_dir / "zf_data_df.csv", sep=" ", index=False)
