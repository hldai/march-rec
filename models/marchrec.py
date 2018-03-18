import numpy as np
import logging


class MarchRec:
    def __init__(self, n_users, n_items, k, n_epoch=20, batch_size=10):
        self.n_users = n_users
        self.n_items = n_items
        self.mean_val = 0
        self.k = k

        self.P = 0.1 * np.random.randn(self.n_users, self.k)
        self.Q = 0.1 * np.random.randn(self.n_items, self.k)

    def fit(self, entries_train, entries_val):
        self.mean_val = np.mean(entries_train[:, 2])

        n_train, n_val = entries_train.shape[0], entries_val.shape[0]
        logging.info('MarchRec. {} training items, {} val items, mean: {}.'.format(n_train, n_val, self.mean_val))
