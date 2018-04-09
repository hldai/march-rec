import pandas as pd
import os
import datetime
import numpy as np
from utils.loggingutils import init_logging
from config import *
import copy


def __enumerate_all_aparams(param_lists):
    params_list = list()
    __dfs_param_lists(param_lists, [], 0, params_list)
    return params_list


def __get_shuffled_train_entries(train_file):
    df_train = pd.read_csv(train_file, header=None)
    entries_train = df_train.as_matrix([0, 1, 2])
    shuffled_idxs = np.arange(0, entries_train.shape[0])
    return entries_train[shuffled_idxs]


def __run_pmf(train_file, val_file, test_file):
    from models.pmf import PMF

    entries_train = __get_shuffled_train_entries(train_file)

    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])

    df_test = pd.read_csv(test_file, header=None, sep='\t')
    entries_test = df_test.as_matrix([0, 1, 2])

    epsilon = 0.01
    lamb = 0.3

    epsilon_list = [0.01, 0.001]
    lamb_list = [0.2, 0.3, 0.4, 0.5]
    params_list = __enumerate_all_aparams([epsilon_list, lamb_list])
    for epsilon, lamb in params_list:
        pmf = PMF(N_USERS + 1, N_ITEMS + 1, k=10, epsilon=epsilon, lamb=lamb, n_epoch=200, batch_size=10)
        pmf.fit(entries_train, entries_val, entries_test)


def __run_biased_svd(train_file, val_file, test_file):
    from models.biasedsvd import BiasedSVD

    entries_train = __get_shuffled_train_entries(train_file)

    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])

    df_test = pd.read_csv(test_file, header=None, sep='\t')
    entries_test = df_test.as_matrix([0, 1, 2])

    lr = 0.001
    lamb = 0.1
    bsvd = BiasedSVD(N_USERS + 1, N_ITEMS + 1, k=10, learning_rate=lr, lamb=lamb, n_epoch=200, batch_size=10)
    bsvd.fit(entries_train, entries_val, entries_test)

    # epsilon_list = [0.01, 0.001]
    # lamb_list = [0.2, 0.3, 0.4, 0.5]
    # params_list = __enumerate_all_aparams([epsilon_list, lamb_list])
    # for epsilon, lamb in params_list:
    #     pmf = PMF(N_USERS + 1, N_ITEMS + 1, k=10, epsilon=epsilon, lamb=lamb, n_epoch=200, batch_size=10)
    #     pmf.fit(entries_train, entries_val, entries_test)


def __run_mlp(train_file, val_file):
    from models.mlp import MLPRec

    mlpr = MLPRec(N_USERS + 1, N_ITEMS + 1, [10, 10], learning_rate=0.1, lamb=0.1, n_epochs=10)

    entries_train = __get_shuffled_train_entries(train_file)
    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])

    mlpr.fit(entries_train, entries_val)


def __run_dmf(train_file, val_file, test_file):
    from models.dmf import DMF

    dmf = DMF()


def __dfs_param_lists(param_lists, cur_params, cur_pos, cur_params_list):
    if cur_pos == len(param_lists):
        cur_params_list.append(cur_params)
        return
    for p in param_lists[cur_pos]:
        # print(cur_params)
        tmp = copy.copy(cur_params)
        tmp.append(p)
        __dfs_param_lists(param_lists, tmp, cur_pos + 1, cur_params_list)


def __run_march(train_file, val_file, test_file):
    from models.marchrec import MarchRec

    entries_train = __get_shuffled_train_entries(train_file)
    df_val = pd.read_csv(val_file, header=None)
    entries_val = df_val.as_matrix([0, 1, 2])
    df_test = pd.read_csv(test_file, header=None, sep='\t')
    entries_test = df_test.as_matrix([0, 1, 2])

    lamb1, lamb2, lamb3 = 0.1, 0.1, 0.1
    lr = 0.001
    batch_size = 10
    n_epoch = 80
    k = 10
    alpha1, alpha2, alpha3 = 0.1, 1, 0.01

    # mr = MarchRec(N_USERS + 1, N_ITEMS + 1, k, n_epoch, batch_size, lr, alpha1, alpha2, alpha3, lamb1, lamb2, lamb3)
    # mr.fit(entries_train, entries_val, entries_test)

    lamb1_list, lamb2_list, lamb3_list = [0.01, 0.1], [0.01, 0.1], [0.01, 0.1]
    lr_list = [0.001]
    alpha1_list, alpha2_list, alpha3_list = [0.01, 0.1, 1], [0.01, 0.1, 1], [0.01, 0.1, 1]
    param_lists = [lr_list, lamb1_list, lamb2_list, lamb3_list, alpha1_list, alpha2_list, alpha3_list]
    params_list = __enumerate_all_aparams(param_lists)

    for params in params_list:
        lr, lamb1, lamb2, lamb3, alpha1, alpha2, alpha3 = params
        if lamb1 == 0.01 and lamb2 == 0.01:
            continue
        mr = MarchRec(N_USERS + 1, N_ITEMS + 1, k, n_epoch, batch_size, lr, alpha1, alpha2, alpha3, lamb1, lamb2, lamb3)
        mr.fit(entries_train, entries_val, entries_test)


# method = 'pmf'
# method = 'biased_svd'
method = 'dmf'

str_today = datetime.date.today().strftime('%y-%m-%d')
init_logging('log/{}-{}.log'.format(method, str_today), to_stdout=True)

split_id = 1
train_file = os.path.join(DATADIR, 'u{}_train.txt'.format(split_id))
val_file = os.path.join(DATADIR, 'u{}_val.txt'.format(split_id))
test_file = os.path.join(DATADIR, 'u{}.test'.format(split_id))

if method == 'pmf':
    __run_pmf(train_file, val_file, test_file)
if method == 'biased_svd':
    __run_biased_svd(train_file, val_file, test_file)
if method == 'dmf':
    __run_dmf(train_file, val_file, test_file)
# __run_mlp(train_file, val_file)
# __run_march(train_file, val_file, test_file)
