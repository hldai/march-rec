import pandas as pd
import os
import datetime
import numpy as np
from utils.loggingutils import init_logging
from config import *


def __get_shuffled_train_entries(train_file):
    df_train = pd.read_csv(train_file, header=None)
    entries_train = df_train.as_matrix([0, 1, 2])
    shuffled_idxs = np.arange(0, entries_train.shape[0])
    return entries_train[shuffled_idxs]


def __run_pmf(train_file, val_file):
    from models.pmf import PMF

    entries_train = __get_shuffled_train_entries(train_file)

    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])

    pmf = PMF(N_USERS + 1, N_ITEMS + 1, k=20, epsilon=0.01, lamb=0.3, n_epoch=200, batch_size=10)
    pmf.fit(entries_train, entries_val)


def __run_mlp(train_file, val_file):
    from models.mlp import MLPRec

    mlpr = MLPRec(N_USERS + 1, N_ITEMS + 1, [10, 10], learning_rate=0.1, lamb=0.1, n_epochs=10)

    entries_train = __get_shuffled_train_entries(train_file)
    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])

    mlpr.fit(entries_train, entries_val)


def __run_march(train_file, val_file):
    from models.marchrec import MarchRec

    entries_train = __get_shuffled_train_entries(train_file)
    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])

    lamb = 0.1
    lr = 0.001
    batch_size = 10
    n_epoch = 50
    k = 10
    mr = MarchRec(N_USERS + 1, N_ITEMS + 1, k, lamb, n_epoch, batch_size, lr)
    mr.fit(entries_train, entries_val)


str_today = datetime.date.today().strftime('%y-%m-%d')
init_logging('log/{}.log'.format(str_today), to_stdout=True)

split_id = 1
train_file = os.path.join(DATADIR, 'u{}_train.txt'.format(split_id))
val_file = os.path.join(DATADIR, 'u{}_val.txt'.format(split_id))

# __run_pmf(train_file, val_file)
# __run_mlp(train_file, val_file)
__run_march(train_file, val_file)
