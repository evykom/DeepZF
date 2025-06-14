from functions import *


def get_label_mat(c_rc_df):
    label_mat = (c_rc_df.filter(items=['A1', 'C1', 'G1', 'T1', 'A2', 'C2', 'G2', 'T2', 'A3', 'C3', 'G3', 'T3'])).values
    return label_mat


def create_input_model(df, res_num):
    if res_num == 36:
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_36_neighbors'])
    elif res_num == 12:
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_12'])
    elif res_num == 7:
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_7_b1h'])
    elif res_num == 4:  # res_num == 4
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_4'])
    else:
        # fall back to generic res_<num> column if present
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df[f'res_{res_num}'])

    return oneHot_C_RC_amino


