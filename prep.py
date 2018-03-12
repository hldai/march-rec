import os
import pandas as pd
import numpy as np
from config import *


def __split_train_data(all_train_file, dst_train_file, dst_val_file):
    val_rate = 0.2
    df = pd.read_csv(all_train_file, sep='\t', header=None)
    n_items = df.shape[0]
    print(n_items, 'items in total,', val_rate, 'will be used for validation')
    n_val = int(n_items * val_rate)
    perm = np.random.permutation(n_items)

    with open(dst_train_file, 'w', encoding='utf-8', newline='\n') as fout:
        df.iloc[perm[n_val:], :].to_csv(fout, header=False, index=False)
    with open(dst_val_file, 'w', encoding='utf-8', newline='\n') as fout:
        df.iloc[perm[:n_val], :].to_csv(fout, header=False, index=False)


split_id = 1
all_train_file = os.path.join(DATADIR, 'u{}.base'.format(split_id))
train_file = os.path.join(DATADIR, 'u{}_train.txt'.format(split_id))
val_file = os.path.join(DATADIR, 'u{}_val.txt'.format(split_id))

__split_train_data(all_train_file, train_file, val_file)
