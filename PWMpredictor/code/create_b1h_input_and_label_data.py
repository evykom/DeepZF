from pathlib import Path
from functions import *

"""This function creates B1H input and label for Transfer learning model:
INPUT DATA: each amino acid is represented by a 1X20 one hot vector therefore
             zinc finger with 12 positions is represented by 1X240 vector
             zinc finger with 7 positions is represented by 1X140 vector and
             we pad each finger to be a 12 positions long therefore final representation is 1x240
 LABEL: the model label is one a position weight matrix (pwm): 3X4 (3 possible positions and 4 DNA nucleotides),
        In our model the pwm is reshaped to  1X12 vector"""



B1H_one_finger_add = "../../Data/PWMpredictor/one_finger_pwm_gt.txt"
file = open(B1H_one_finger_add)  # open data file

mat_l = []
prot_7_res_seq_l = []  # 7 residuals list
prot_4_res_seq_l = []  # 4 residuals list
prot_name_l = []       # protein names
zf_index_l = []        # finger indices

lines = file.readlines()
for i in range(0, len(lines)-2, 8):
    mat_l.append(lines[i+3: i+7])
    prot_4_res_seq_l.append(lines[i+1].rstrip())
    prot_7_res_seq_l.append(lines[i+2].rstrip())

    name = lines[i].rstrip()
    # name format example: "10G_SDM.zf.F2"
    try:
        prot, _, idx = name.split(".zf.")
        idx = int(idx.lstrip("F"))
    except ValueError:
        prot, idx = name, -1
    prot_name_l.append(prot)
    zf_index_l.append(idx)

file.close()

"create pwm matrix (label)"
mat_pd = pd.DataFrame(mat_l).drop(columns=4)
mat_pd = mat_pd.applymap(lambda x: x[4: -1])
mat_pd['all'] = mat_pd[0] + ' ' + mat_pd[1] + ' ' + mat_pd[2] + ' ' + mat_pd[3]
mat_pd = mat_pd.applymap(lambda x: np.fromstring(x, dtype=float, sep=' '))
pwm = np.stack(mat_pd['all'])
reorder_index = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
pwm = pwm[:, reorder_index]

"create input data_frame: one hot encoding without amino acid X:"
"each amino acid is a binary 20 length vector"
string = 'XXXXX'
prot_df = pd.DataFrame({
    'prot_name': prot_name_l,
    'zf_index': zf_index_l,
    'res_4': prot_4_res_seq_l,
    'res_7': prot_7_res_seq_l,
})
prot_df['res_12'] = prot_df['res_7'].apply(lambda s: string + s)

def add_neighbor_feature(df: pd.DataFrame,
                         pad_char: str = 'X', pad_len: int = 12) -> pd.DataFrame:
    df = df.sort_values(['prot_name', 'zf_index']).reset_index(drop=True)
    df['_prev_seq'] = df.groupby('prot_name')['res_12'].shift(1)
    df['_prev_idx'] = df.groupby('prot_name')['zf_index'].shift(1)
    df['_next_seq'] = df.groupby('prot_name')['res_12'].shift(-1)
    df['_next_idx'] = df.groupby('prot_name')['zf_index'].shift(-1)

    def _join(row):
        prev_ok = pd.notna(row['_prev_seq']) and row['zf_index'] - 1 == row['_prev_idx']
        next_ok = pd.notna(row['_next_seq']) and row['zf_index'] + 1 == row['_next_idx']
        prev = row['_prev_seq'] if prev_ok else pad_char * pad_len
        nxt = row['_next_seq'] if next_ok else pad_char * pad_len
        return f"{prev}{row['res_12']}{nxt}"

    df['res_36_neighbors'] = df.apply(_join, axis=1)
    return df.drop(columns=['_prev_seq', '_prev_idx', '_next_seq', '_next_idx'])

prot_df = add_neighbor_feature(prot_df)
prot_7_res_df = prot_df[['res_7']].copy()
prot_12_res_df = prot_df['res_12']

"""Create additional padded representations for longer sequences.
The original B1H library contains seven informative residues of the
finger.  For the transfer learning models the sequences were padded
with five ``X`` characters at the N‑terminus to generate a 12 residue
representation.  For some experiments we would also like to supply
additional flanking positions around the canonical twelve residues.
Since the B1H library does not contain this information we simply pad
the sequence symmetrically with ``X`` characters."""

# Padding sizes: number of additional residues on each side
flank_sizes = {16: 2, 24: 6, 36: 12}
# Create padded sequences for the requested lengths
padded_seqs = {}
for length, flank in flank_sizes.items():
    padded_seqs[length] = prot_12_res_df.apply(
        lambda s: "X" * flank + s + "X" * flank)

prot_4_res_df = pd.DataFrame(prot_4_res_seq_l)


"save model input and label of data including amino acid X"
"amino acid X is encoded as a 20 length vector with probability 1/20"
one_hot_12res = oneHot_Amino_acid_vec(prot_12_res_df)
one_hot_extended = {}
for length, seqs in padded_seqs.items():
    one_hot_extended[length] = oneHot_Amino_acid_vec(seqs)
one_hot_36neigh = oneHot_Amino_acid_vec(prot_df['res_36_neighbors'])

out_dir = Path('../../Data/PWMpredictor/new_data')
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / 'ground_truth_b1h_pwm_12res.npy', pwm)
np.save(out_dir / 'onehot_encoding_b1h_12res.npy', one_hot_12res)

for length, one_hot in one_hot_extended.items():
    np.save(out_dir / f'ground_truth_b1h_pwm_{length}res.npy', pwm)
    np.save(out_dir / f'onehot_encoding_b1h_{length}res.npy', one_hot)

np.save(out_dir / 'ground_truth_b1h_pwm_36neigh.npy', pwm)
np.save(out_dir / 'onehot_encoding_b1h_36neigh.npy', one_hot_36neigh)

""" one hot encoding for sequences without amino acid X"""

def find_X_amino_acid_index(prot_pd):
    """There are some protein sequences with X as amino acid, this function finds this indexes"""
    x_index_l = []
    for i in range(prot_pd.shape[0]):
        if "X" in prot_pd[0][i]:
            x_index_l.append(i)
    return x_index_l


x_index_4res_l = find_X_amino_acid_index(prot_4_res_df)
x_index_7res_l = find_X_amino_acid_index(prot_7_res_df)

# drop from data dataframe the protein sequences with amino acid X
prot_4_res_df.drop(x_index_4res_l, inplace=True)
prot_7_res_df.drop(x_index_7res_l, inplace=True)

"find one hot representation: each protein representd by on hot vector"
one_hot_4res = oneHot_Amino_acid_vec(prot_4_res_df[0])
one_hot_7res = oneHot_Amino_acid_vec(prot_7_res_df[0])

"update pwm matrix"
pwm_4res = np.delete(pwm, x_index_4res_l, axis=0)
pwm_7res = np.delete(pwm, x_index_7res_l, axis=0)


np.save(out_dir / 'ground_truth_b1h_pwm_4res.npy', pwm_4res)
np.save(out_dir / 'ground_truth_b1h_pwm_7res.npy', pwm_7res)
np.save(out_dir / 'onehot_encoding_b1h_4res.npy', one_hot_4res)
np.save(out_dir / 'onehot_encoding_b1h_7res.npy', one_hot_7res)

