
from functions import *
import pandas as pd


c_rc_df   = pd.read_csv("../../data/c_rc_df.csv")     
zf_data_df = pd.read_csv("../../data/zf_data_df.csv") 

# ------------------------------------------------------------------
#  Additional representations with flanking residues
# ------------------------------------------------------------------

def pad_with_x(seq, flank):
    """Pad sequence with ``X`` characters on both sides."""
    return "X" * flank + seq + "X" * flank


def extract_with_flank(row, flank):
    """Extract the finger with additional flanking residues from the
    original protein sequence."""
    start = int(row['zf_indx_start'])
    end = int(row['zf_indx_end'])
    prot_seq = row['prot_seq']
    left = max(start - flank, 0)
    right = min(end + flank, len(prot_seq))
    return prot_seq[left:right]


flank_sizes = {16: 2, 24: 6, 36: 12}

# Create padded versions for the c_rc dataframe (padding with X)
for length, flank in flank_sizes.items():
    col = f'res_{length}'
    c_rc_df[col] = c_rc_df['res_12'].apply(lambda s: pad_with_x(s, flank))

# Create versions with real flanking residues for the zf_data dataframe
for length, flank in flank_sizes.items():
    col = f'res_{length}'
    zf_data_df[col] = zf_data_df.apply(lambda r: extract_with_flank(r, flank), axis=1)


"one hot encoding"
one_hot_c_rc_4res = oneHot_Amino_acid_vec(c_rc_df['res_4'])
one_hot_c_rc_7res = oneHot_Amino_acid_vec(c_rc_df['res_7'])
one_hot_c_rc_7b1h_res = oneHot_Amino_acid_vec(c_rc_df['res_7_b1h'])
one_hot_c_rc_12res = oneHot_Amino_acid_vec(c_rc_df['res_12'])
one_hot_c_rc_extended = {}
for length in flank_sizes:
    col = f'res_{length}'
    one_hot_c_rc_extended[length] = oneHot_Amino_acid_vec(c_rc_df[col])

one_hot_zf_extended = {}
for length in flank_sizes:
    col = f'res_{length}'
    one_hot_zf_extended[length] = oneHot_Amino_acid_vec(zf_data_df[col])

"labels: the pwm is the same to all sequence residuals"
pwm = (c_rc_df.filter(items=['A1', 'C1', 'G1', 'T1', 'A2', 'C2', 'G2', 'T2', 'A3', 'C3', 'G3', 'T3'])).values
"Savings"
path = '/Transfer_learning/data_labels/'
np.save(path + 'ground_truth_c_rc', pwm)
np.save(path + 'onehot_encoding_c_rc_4res', one_hot_c_rc_4res)
np.save(path + 'onehot_encoding_c_rc_7res', one_hot_c_rc_7res)
np.save(path + 'onehot_encoding_c_rc_7b1hres', one_hot_c_rc_7b1h_res)
np.save(path + 'onehot_encoding_c_rc_12res', one_hot_c_rc_12res)
for length, one_hot in one_hot_c_rc_extended.items():
    np.save(path + f'onehot_encoding_c_rc_{length}res', one_hot)
for length, one_hot in one_hot_zf_extended.items():
    np.save(path + f'onehot_encoding_zf_{length}res', one_hot)
c_rc_df.to_csv(path + 'c_rc_df.csv', sep=' ', index=False)
zf_data_df.to_csv(path + 'zf_data_df.csv', sep=' ', index=False)
