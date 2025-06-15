#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_b1h_input_and_label_data.py
----------------------------------
• Parse the B1H “one-finger” file (`one_finger_pwm_gt.txt`)
• Build input sets:
    – 4-aa core                       (X-free only)
    – 7-aa core                       (X-free only)
    – 12-aa core  (5 × “X” + 7)       (all sequences)
    – 20-aa neighbour window 4+12+4   (all sequences)
    – 26-aa neighbour window 7+12+7   (all sequences)
    – 36-aa neighbour window 12+12+12 (all sequences)
• One-hot encode every window (20-long vector per residue, “X” = 0.05 each)
• Save .npy files under ../../Data/PWMpredictor/new_data/
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
from functions import *            # oneHot_Amino_acid_vec, etc.

# ----------------------------------------------------------------------
# Read raw B1H file
# ----------------------------------------------------------------------
B1H_TXT = "../../Data/PWMpredictor/one_finger_pwm_gt.txt"

mat_l, prot_7_res_seq_l, prot_4_res_seq_l = [], [], []
prot_name_l, zf_index_l = [], []

with open(B1H_TXT) as fh:
    lines = fh.readlines()

for i in range(0, len(lines) - 2, 8):
    # four PWM rows (A, C, G, T)
    mat_l.append(lines[i + 3 : i + 7])

    # residue strings
    prot_4_res_seq_l.append(lines[i + 1].rstrip())
    prot_7_res_seq_l.append(lines[i + 2].rstrip())

    # protein ID and finger index (e.g. “10G_SDM.zf.F2”)
    name = lines[i].rstrip()
    try:
        prot, _, idx = name.split(".zf.")
        idx = int(idx.lstrip("F"))
    except ValueError:
        prot, idx = name, -1
    prot_name_l.append(prot)
    zf_index_l.append(idx)

# ----------------------------------------------------------------------
# PWM (n_fingers × 12)
# ----------------------------------------------------------------------
mat_pd = pd.DataFrame(mat_l).applymap(lambda s: s[4:-1])      # strip “PWM:” and \n
mat_pd["all"] = mat_pd[0] + " " + mat_pd[1] + " " + mat_pd[2] + " " + mat_pd[3]
pwm = np.stack(mat_pd["all"].map(lambda s: np.fromstring(s, sep=" ")))
pwm = pwm[:, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]          # reorder to A1,C1,G1,T1,…

# ----------------------------------------------------------------------
# Build residue DataFrame (4-, 7-, 12-aa cores)
# ----------------------------------------------------------------------
pad5 = "XXXXX"                                                # 5 × “X” → 12-aa core

prot_df = pd.DataFrame({
    "prot_name": prot_name_l,
    "zf_index" : zf_index_l,
    "res_4"    : prot_4_res_seq_l,
    "res_7"    : prot_7_res_seq_l,
})
prot_df["res_12"] = prot_df["res_7"].apply(lambda s: pad5 + s)

# ----------------------------------------------------------------------
# Neighbour-window helpers
# ----------------------------------------------------------------------
def add_neighbour_window(df: pd.DataFrame,
                         k_left: int,
                         k_right: int,
                         pad_char: str = "X") -> pd.Series:
    """
    Build a window:   left(k_left) + current 12 + right(k_right)

    * If the previous / next finger is contiguous (index ±1) – take the
      required number of residues from that neighbour.
    * Otherwise, pad with `pad_char`.
    """
    pad_left  = pad_char * k_left
    pad_right = pad_char * k_right

    df = df.sort_values(["prot_name", "zf_index"]).reset_index(drop=True)
    prev_seq = df.groupby("prot_name")["res_12"].shift(1)
    prev_idx = df.groupby("prot_name")["zf_index"].shift(1)
    next_seq = df.groupby("prot_name")["res_12"].shift(-1)
    next_idx = df.groupby("prot_name")["zf_index"].shift(-1)

    out = []
    for row, p_seq, p_idx, n_seq, n_idx in zip(
            df.itertuples(index=False), prev_seq, prev_idx, next_seq, next_idx):

        prev_ok = pd.notna(p_seq) and row.zf_index - 1 == p_idx
        next_ok = pd.notna(n_seq) and row.zf_index + 1 == n_idx

        left  = (p_seq[-k_left:] if (k_left and prev_ok) else pad_left)
        right = (n_seq[:k_right] if (k_right and next_ok) else pad_right)
        out.append(f"{left}{row.res_12}{right}")

    return pd.Series(out, index=df.index)

# Create neighbour windows
prot_df["res_20_neighbors"] = add_neighbour_window(prot_df, 4, 4)    # 4+12+4
prot_df["res_26_neighbors"] = add_neighbour_window(prot_df, 7, 7)    # 7+12+7
prot_df["res_36_neighbors"] = add_neighbour_window(prot_df, 12, 12)  # 12+12+12

# ----------------------------------------------------------------------
# Remove rows containing “X” for 4- and 7-aa sets
# ----------------------------------------------------------------------
x_mask_4 = prot_df["res_4"].str.contains("X")
x_mask_7 = prot_df["res_7"].str.contains("X")

one_hot_4 = oneHot_Amino_acid_vec(prot_df.loc[~x_mask_4, "res_4"])
one_hot_7 = oneHot_Amino_acid_vec(prot_df.loc[~x_mask_7, "res_7"])

pwm_4 = pwm[~x_mask_4]
pwm_7 = pwm[~x_mask_7]

# ----------------------------------------------------------------------
# One-hot encodings for ALL sequences
# ----------------------------------------------------------------------
one_hot_12       = oneHot_Amino_acid_vec(prot_df["res_12"])
one_hot_20neigh  = oneHot_Amino_acid_vec(prot_df["res_20_neighbors"])
one_hot_26neigh  = oneHot_Amino_acid_vec(prot_df["res_26_neighbors"])
one_hot_36neigh  = oneHot_Amino_acid_vec(prot_df["res_36_neighbors"])

# ----------------------------------------------------------------------
# Save to disk
# ----------------------------------------------------------------------
out_dir = Path("../../Data/PWMpredictor/new_data")
out_dir.mkdir(parents=True, exist_ok=True)

# 12-aa core
np.save(out_dir / "ground_truth_b1h_pwm_12res.npy", pwm)
np.save(out_dir / "onehot_encoding_b1h_12res.npy",  one_hot_12)

# 20-aa neighbour (4+12+4)
np.save(out_dir / "ground_truth_b1h_pwm_20neigh.npy", pwm)
np.save(out_dir / "onehot_encoding_b1h_20neigh.npy",  one_hot_20neigh)

# 26-aa neighbour (7+12+7)
np.save(out_dir / "ground_truth_b1h_pwm_26neigh.npy", pwm)
np.save(out_dir / "onehot_encoding_b1h_26neigh.npy",  one_hot_26neigh)

# 36-aa neighbour (12+12+12)
np.save(out_dir / "ground_truth_b1h_pwm_36neigh.npy", pwm)
np.save(out_dir / "onehot_encoding_b1h_36neigh.npy",  one_hot_36neigh)

# X-free 4/7 sets
np.save(out_dir / "ground_truth_b1h_pwm_4res.npy", pwm_4)
np.save(out_dir / "onehot_encoding_b1h_4res.npy",  one_hot_4)

np.save(out_dir / "ground_truth_b1h_pwm_7res.npy", pwm_7)
np.save(out_dir / "onehot_encoding_b1h_7res.npy",  one_hot_7)
