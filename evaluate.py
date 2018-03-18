import numpy as np


def rmse_bin(pos_entries, y_pred):
    n_vals = y_pred.shape[0] * y_pred.shape[1]
    user_idxs = pos_entries[:, 0]
    item_idxs = pos_entries[:, 1]
    y_pred_pos = y_pred[user_idxs, item_idxs]
    err = np.sum((1 - y_pred_pos) ** 2) + np.sum(y_pred ** 2) - np.sum(y_pred_pos ** 2)
    return np.sqrt(err) / np.sqrt(n_vals)
