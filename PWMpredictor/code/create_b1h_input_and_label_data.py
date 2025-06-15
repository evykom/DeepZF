#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_b1h_input_and_label_data.py
----------------------------------
• Reads the original B1H one-finger file (`one_finger_pwm_gt.txt`)
• Builds four input sets – 4-aa, 7-aa, 12-aa (5×“X” + 7) and 36-aa “neighbour”
  window (prev + curr + next finger, X-padded when absent)
• Generates one-hot encodings (20-length vector per residue, “X” = uniform 0.05)
• Writes .npy files under ../../Data/PWMpredictor/new_data/

‼️  No longer creates the extra 16/24/36-padded windows.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
from functions import *          # oneHot_Amino_acid_vec, etc.

# ------------------------------------------------------------------
# Read the raw B1H file
# ------------------------------------------------------------------
B1H_TXT = "../../Data/PWMpredictor/one_finger_pwm_gt.txt"

mat_l, prot_7_res_seq_l, prot_4_res_seq_l = [], [], []
prot_name_l, zf_index_l = [], []

with open(B1H_TXT) as fh:
    lines = fh.readlines()

for i in range(0, len(lines) - 2, 8):
    # 4 PWM rows (A,C,G,T)
    mat_l.append(lines[i + 3 : i + 7])

    # residue strings
    prot_4_res_seq_l.append(lines[i + 1].rstrip())
    prot_7_res_seq_l.append(lines[i + 2].rstrip())

    # protein id and finger index (e.g. “10G_SDM.zf.F2”)
    name = lines[i].rstrip()
    try:
        prot, _, idx = name.split(".zf.")
        idx = int(idx.lstrip("F"))
    except ValueError:
        prot, idx = name, -1
    prot_name_l.append(prot)
    zf_index_l.append(idx)

# ------------------------------------------------------------------
# Assemble PWM label matrix  (n_fingers × 12)
# ------------------------------------------------------------------
mat_pd = pd.DataFrame(mat_l)
mat_pd = mat_pd.applymap(lambda s: s[4:-1])           # strip “PWM:” and \n
mat_pd["all"] = mat_pd[0] + " " + mat_pd[1] + " " + mat_pd[2] + " " + mat_pd[3]
pwm = np.stack(mat_pd["all"].map(lambda s: np.fromstring(s, sep=" ")))

# reorder → A1,C1,G1,T1,A2,…,T3
reorder = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
pwm = pwm[:, reorder]

# ------------------------------------------------------------------
# Build residue DataFrame  (4-, 7-, 12- and 36-aa)
# ------------------------------------------------------------------
pad5 = "XXXXX"                                          # five X’s → 12-aa core

prot_df = pd.DataFrame(
    {
        "prot_name": prot_name_l,
        "zf_index" : zf_index_l,
        "res_4"    : prot_4_res_seq_l,
        "res_7"    : prot_7_res_seq_l,
    }
)
prot_df["res_12"] = prot_df["res_7"].apply(lambda s: pad5 + s)


def add_neighbor_feature(df: pd.DataFrame,
                         pad_char: str = "X", pad_len: int = 12) -> pd.DataFrame:
    """Return df with a new column ‘res_36_neighbors’ = prev + curr + next."""
    df = df.sort_values(["prot_name", "zf_index"]).reset_index(drop=True)

    df["_prev"]     = df.groupby("prot_name")["res_12"].shift(1)
    df["_prev_idx"] = df.groupby("prot_name")["zf_index"].shift(1)
    df["_next"]     = df.groupby("prot_name")["res_12"].shift(-1)
    df["_next_idx"] = df.groupby("prot_name")["zf_index"].shift(-1)

    def glue(r):
        prev_ok = pd.notna(r["_prev"]) and r["zf_index"] - 1 == r["_prev_idx"]
        next_ok = pd.notna(r["_next"]) and r["zf_index"] + 1 == r["_next_idx"]
        prev = r["_prev"] if prev_ok else pad_char * pad_len
        nxt  = r["_next"] if next_ok else pad_char * pad_len
        return f"{prev}{r['res_12']}{nxt}"

    df["res_36_neighbors"] = df.apply(glue, axis=1)    # exactly 36 aa
    return df.drop(columns=["_prev", "_prev_idx", "_next", "_next_idx"])


prot_df = add_neighbor_feature(prot_df)

# ------------------------------------------------------------------
# Build **X-free** subsets (drop rows containing “X”)
# ------------------------------------------------------------------
prot_4_df = pd.DataFrame({"res_4": prot_4_res_seq_l})
prot_7_df = prot_df[["res_7"]].copy()

rows_with_X = lambda df: df.index[df.iloc[:, 0].str.contains("X")].tolist()

x_idx_4 = rows_with_X(prot_4_df)
x_idx_7 = rows_with_X(prot_7_df)

prot_4_df.drop(index=x_idx_4, inplace=True)
prot_7_df.drop(index=x_idx_7, inplace=True)

# ------------------------------------------------------------------
# One-hot encodings
# ------------------------------------------------------------------
one_hot_4        = oneHot_Amino_acid_vec(prot_4_df.iloc[:, 0])
one_hot_7        = oneHot_Amino_acid_vec(prot_7_df.iloc[:, 0])
one_hot_12       = oneHot_Amino_acid_vec(prot_df["res_12"])
one_hot_36neigh  = oneHot_Amino_acid_vec(prot_df["res_36_neighbors"])

# ------------------------------------------------------------------
# PWM subsets matching X-free 4/7 splits
# ------------------------------------------------------------------
pwm_4 = np.delete(pwm, x_idx_4, axis=0)
pwm_7 = np.delete(pwm, x_idx_7, axis=0)

# ------------------------------------------------------------------
# Save everything
# ------------------------------------------------------------------
out_dir = Path("../../Data/PWMpredictor/new_data")
out_dir.mkdir(parents=True, exist_ok=True)

# core 12-aa set (all sequences)
np.save(out_dir / "ground_truth_b1h_pwm_12res.npy", pwm)
np.save(out_dir / "onehot_encoding_b1h_12res.npy",  one_hot_12)

# neighbour 36-aa set (all sequences)
np.save(out_dir / "ground_truth_b1h_pwm_36neigh.npy", pwm)
np.save(out_dir / "onehot_encoding_b1h_36neigh.npy",  one_hot_36neigh)

# X-free 4/7 sets
np.save(out_dir / "ground_truth_b1h_pwm_4res.npy", pwm_4)
np.save(out_dir / "onehot_encoding_b1h_4res.npy",  one_hot_4)

np.save(out_dir / "ground_truth_b1h_pwm_7res.npy", pwm_7)
np.save(out_dir / "onehot_encoding_b1h_7res.npy",  one_hot_7)
