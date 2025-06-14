from pathlib import Path
import numpy as np
import pandas as pd
from functions import *            # oneHot_Amino_acid_vec, etc.

# ------------------------------------------------------------------
#  Load the raw tables
# ------------------------------------------------------------------
# white-space (one or more spaces / tabs) is the delimiter
c_rc_df   = pd.read_csv("../../Data/PWMpredictor/c_rc_df.csv",
                        sep=r"\s+", engine="python")

zf_data_df = pd.read_csv("../../Data/PWMpredictor/zf_data_df.csv",
                         sep=r"\s+", engine="python")


# column names in each table
CRC_PROT_COL = "UniProt_ID"
CRC_IDX_COL  = "ZF_index"

ZF_PROT_COL  = "prot_name_id"
ZF_IDX_COL   = "zf_index"

# ------------------------------------------------------------------
#  Build 36-aa neighbour window for the zf_data table
# ------------------------------------------------------------------
def add_neighbor_feature(df: pd.DataFrame,
                         prot_col: str,
                         idx_col:  str,
                         pad_char: str = "X",
                         pad_len:  int = 12) -> pd.DataFrame:
    """Create df['res_36_neighbors'] = prev + curr + next finger (36 aa)."""
    df = df.sort_values([prot_col, idx_col]).reset_index(drop=True)

    df["_prev_seq"] = df.groupby(prot_col)["res_12"].shift(1)
    df["_prev_idx"] = df.groupby(prot_col)[idx_col].shift(1)
    df["_next_seq"] = df.groupby(prot_col)["res_12"].shift(-1)
    df["_next_idx"] = df.groupby(prot_col)[idx_col].shift(-1)

    def _join(row):
        prev_ok = pd.notna(row["_prev_seq"]) and row[idx_col] - 1 == row["_prev_idx"]
        next_ok = pd.notna(row["_next_seq"]) and row[idx_col] + 1 == row["_next_idx"]
        prev = row["_prev_seq"] if prev_ok else pad_char * pad_len
        nxt  = row["_next_seq"] if next_ok else pad_char * pad_len
        return f"{prev}{row['res_12']}{nxt}"

    df["res_36_neighbors"] = df.apply(_join, axis=1)
    return df.drop(columns=["_prev_seq", "_prev_idx", "_next_seq", "_next_idx"])

zf_data_df = add_neighbor_feature(zf_data_df, ZF_PROT_COL, ZF_IDX_COL)

# ------------------------------------------------------------------
#  Copy the neighbour window into the c_rc table
# ------------------------------------------------------------------
c_rc_df = c_rc_df.merge(
    zf_data_df[[ZF_PROT_COL, ZF_IDX_COL, "res_36_neighbors"]],
    left_on=[CRC_PROT_COL, CRC_IDX_COL],
    right_on=[ZF_PROT_COL, ZF_IDX_COL],
    how="left"
)
c_rc_df["res_36_neighbors"].fillna(c_rc_df["res_12"], inplace=True)
c_rc_df.drop(columns=[ZF_PROT_COL, ZF_IDX_COL], inplace=True)

# ------------------------------------------------------------------
#  Fixed-length windows (padding or real flanks)
# ------------------------------------------------------------------
def pad_with_x(seq: str, flank: int) -> str:
    return f"{'X'*flank}{seq}{'X'*flank}"

def extract_with_flank(row, flank: int) -> str:
    start, end = int(row["zf_indx_start"]), int(row["zf_indx_end"])
    left  = max(start - flank, 0)
    right = min(end + flank, len(row["prot_seq"]))
    return row["prot_seq"][left:right]

flank_sizes = {16: 2, 24: 6, 36: 12}

for length, flank in flank_sizes.items():
    c_rc_df[f"res_{length}"] = c_rc_df["res_12"].apply(pad_with_x, flank=flank)
    zf_data_df[f"res_{length}"] = zf_data_df.apply(extract_with_flank, flank=flank,
                                                   axis=1)

# ------------------------------------------------------------------
#  One-hot encodings
# ------------------------------------------------------------------
one_hot_c_rc = {
    "4"      : oneHot_Amino_acid_vec(c_rc_df["res_4"]),
    "7"      : oneHot_Amino_acid_vec(c_rc_df["res_7"]),
    "7b1h"   : oneHot_Amino_acid_vec(c_rc_df["res_7_b1h"]),
    "12"     : oneHot_Amino_acid_vec(c_rc_df["res_12"]),
    "36neigh": oneHot_Amino_acid_vec(c_rc_df["res_36_neighbors"]),
}
one_hot_zf = {
    "36neigh": oneHot_Amino_acid_vec(zf_data_df["res_36_neighbors"]),
}
for length in flank_sizes:
    one_hot_c_rc[str(length)] = oneHot_Amino_acid_vec(c_rc_df[f"res_{length}"])
    one_hot_zf[str(length)]   = oneHot_Amino_acid_vec(zf_data_df[f"res_{length}"])

# ------------------------------------------------------------------
#  PWM labels
# ------------------------------------------------------------------
pwm_cols = ['A1','C1','G1','T1','A2','C2','G2','T2','A3','C3','G3','T3']
pwm = c_rc_df[pwm_cols].values

# ------------------------------------------------------------------
#  Save outputs
# ------------------------------------------------------------------
out_dir = Path("../../Data/PWMpredictor/new_data")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "ground_truth_c_rc.npy", pwm)

for key, arr in one_hot_c_rc.items():
    np.save(out_dir / f"onehot_encoding_c_rc_{key}.npy", arr)
for key, arr in one_hot_zf.items():
    np.save(out_dir / f"onehot_encoding_zf_{key}.npy", arr)

c_rc_df.to_csv(out_dir / "c_rc_df.csv", sep=" ", index=False)
zf_data_df.to_csv(out_dir / "zf_data_df.csv", sep=" ", index=False)
