import pandas as pd
import os
import datetime
from models.pmf import PMF
from utils.loggingutils import init_logging
from config import *


def __run_pmf(train_file, val_file):
    df_train = pd.read_csv(train_file, header=None)
    items_train = df_train.as_matrix([0, 1, 2])
    df_train = pd.read_csv(val_file, header=None)
    items_val = df_train.as_matrix([0, 1, 2])

    pmf = PMF()
    pmf.fit(items_train, items_val)


str_today = datetime.date.today().strftime('%y-%m-%d')
init_logging('log/{}.log'.format(str_today), to_stdout=True)

split_id = 1
train_file = os.path.join(DATADIR, 'u{}_train.txt'.format(split_id))
val_file = os.path.join(DATADIR, 'u{}_val.txt'.format(split_id))
__run_pmf(train_file, val_file)
