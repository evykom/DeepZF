from pathlib import Path
import numpy as np
import pandas as pd
from functions import *        # brings in oneHot_Amino_acid_vec, etc.

# ------------------------------------------------------------------
#  Load input tables
# ------------------------------------------------------------------
c_rc_df   = pd.read_csv("../../data/PWMpredictor/c_rc_df.csv")
zf_data_df = pd.read_csv("../../data/PWMpredictor/zf_data_df.csv")

# ------------------------------------------------------------------
#  36-residue representation built from neighbouring fingers
# ------------------------------------------------------------------
def add_neighbor_feature(df: pd.DataFrame,
                         prot_col: str = "prot_name_id",
                         idx_col:  str = "zf_index",
                         pad_char: str = "X",
                         pad_len:  int = 12) -> pd.DataFrame:
    """
    Add column 'res_36_neighbors' = prev-curr-next finger (12 aa each).
    Missing neighbors are padded with X…X so every string is 36 aa long.
    """
    df = df.sort_values([prot_col, idx_col]).reset_index(drop=True)

    # previous / next finger within each protein
    df["_prev_res12"] = df.groupby(prot_col)["res_12"].shift(1)
    df["_next_res12"] = df.groupby(prot_col)["res_12"].shift(-1)
    df["_prev_idx"]   = df.groupby(prot_col)[idx_col].shift(1)
    df["_next_idx"]   = df.groupby(prot_col)[idx_col].shift(-1)

    def _concat(row):
        prev_ok = pd.notna(row["_prev_res12"]) and row[idx_col] - 1 == row["_prev_idx"]
        next_ok = pd.notna(row["_next_res12"]) and row[idx_col] + 1 == row["_next_idx"]
        prev_seq = row["_prev_res12"] if prev_ok else pad_char * pad_len
        next_seq = row["_next_res12"] if next_ok else pad_char * pad_len
        return f"{prev_seq}{row['res_12']}{next_seq}"

    df["res_36_neighbors"] = df.apply(_concat, axis=1)
    return df.drop(columns=["_prev_res12", "_next_res12", "_prev_idx", "_next_idx"])

# Build neighbour column in the ZF-protein table
zf_data_df = add_neighbor_feature(zf_data_df, "prot_name_id", "zf_index")

# Copy that column into c_rc_df (falls back to the core 12-mer if no match)
c_rc_df = c_rc_df.merge(
    zf_data_df[["prot_name_id", "zf_index", "res_36_neighbors"]],
    left_on=["UniProt_ID", "ZF_index"],
    right_on=["prot_name_id", "zf_index"],
    how="left"
)
c_rc_df["res_36_neighbors"].fillna(c_rc_df["res_12"], inplace=True)
c_rc_df.drop(columns=["prot_name_id", "zf_index"], inplace=True)

# ------------------------------------------------------------------
#  Additional fixed-window representations (synthetic or real flanks)
# ------------------------------------------------------------------
def pad_with_x(seq: str, flank: int) -> str:
    """Return X…X + seq + X…X."""
    return f"{'X'*flank}{seq}{'X'*flank}"

def extract_with_flank(row: pd.Series, flank: int) -> str:
    start, end = int(row["zf_indx_start"]), int(row["zf_indx_end"])
    prot_seq   = row["prot_seq"]
    left  = max(start - flank, 0)
    right = min(end + flank, len(prot_seq))
    return prot_seq[left:right]

flank_sizes = {16: 2, 24: 6, 36: 12}

# c_rc_df – pad with 'X'
for length, flank in flank_sizes.items():
    c_rc_df[f"res_{length}"] = c_rc_df["res_12"].apply(pad_with_x, flank=flank)

# zf_data_df – take real neighbours from the protein sequence
for length, flank in flank_sizes.items():
    zf_data_df[f"res_{length}"] = zf_data_df.apply(extract_with_flank, flank=flank, axis=1)

# ------------------------------------------------------------------
#  One-hot encodings
# ------------------------------------------------------------------
one_hot_c_rc = {
    "4":     oneHot_Amino_acid_vec(c_rc_df["res_4"]),
    "7":     oneHot_Amino_acid_vec(c_rc_df["res_7"]),
    "7b1h":  oneHot_Amino_acid_vec(c_rc_df["res_7_b1h"]),
    "12":    oneHot_Amino_acid_vec(c_rc_df["res_12"]),
    "36neigh": oneHot_Amino_acid_vec(c_rc_df["res_36_neighbors"])
}
for length in flank_sizes:
    one_hot_c_rc[str(length)] = oneHot_Amino_acid_vec(c_rc_df[f"res_{length}"])

one_hot_zf = {
    "36neigh": oneHot_Amino_acid_vec(zf_data_df["res_36_neighbors"])
}
for length in flank_sizes:
    one_hot_zf[str(length)] = oneHot_Amino_acid_vec(zf_data_df[f"res_{length}"])

# ------------------------------------------------------------------
#  PWM labels (same for every representation)
# ------------------------------------------------------------------
pwm_cols = ['A1','C1','G1','T1','A2','C2','G2','T2','A3','C3','G3','T3']
pwm = c_rc_df[pwm_cols].values

# ------------------------------------------------------------------
#  Save everything
# ------------------------------------------------------------------
out_dir = Path("../../data/PWMpredictor/new_data")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "ground_truth_c_rc.npy", pwm)

for key, arr in one_hot_c_rc.items():
    np.save(out_dir / f"onehot_encoding_c_rc_{key}.npy", arr)

for key, arr in one_hot_zf.items():
    np.save(out_dir / f"onehot_encoding_zf_{key}.npy", arr)

c_rc_df.to_csv(out_dir / "c_rc_df.csv", sep=" ", index=False)
zf_data_df.to_csv(out_dir / "zf_data_df.csv", sep=" ", index=False)
