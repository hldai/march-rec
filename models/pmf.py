import numpy as np
import logging


class PMF:
    def __init__(self, n_users, n_items, k=10, epsilon=0.1, lamb=0.1, momentum=0.8, n_epoch=20, batch_size=10):
        self.k = k
        self.epsilon = epsilon
        self.lamb = lamb
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.mean_val = 0
        self.n_users = n_users
        self.n_items = n_items
        self.P = 0.1 * np.random.randn(n_users, self.k)
        self.Q = 0.1 * np.random.randn(n_items, self.k)

    def fit(self, entries_train, entries_val, entries_test):
        self.mean_val = np.mean(entries_train[:, 2])

        n_train, n_val, n_test = entries_train.shape[0], entries_val.shape[0], entries_test.shape[0]
        logging.info('PMF. {} training items, {} val items, {} test items, mean: {}.'.format(
            n_train, n_val, n_test, self.mean_val))

        P_inc = np.zeros((self.n_users, self.k))
        Q_inc = np.zeros((self.n_items, self.k))
        all_user_idxs_train = entries_train[:, 0]
        all_item_idxs_train = entries_train[:, 1]

        n_batches = int(n_train / self.batch_size)
        for it in range(self.n_epoch):
            for batch_idx in range(n_batches):
                indices = np.arange(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))

                user_idxs = np.array(entries_train[indices, 0], dtype='int32')
                item_idxs = np.array(entries_train[indices, 1], dtype='int32')

                pred = np.sum(self.P[user_idxs, :] * self.Q[item_idxs, :], axis=1)
                errs = pred + self.mean_val - entries_train[indices, 2]
                errs = errs.reshape((-1, 1))

                # Compute gradients
                grad_p = 2 * errs * self.Q[item_idxs, :] + self.lamb * self.P[user_idxs, :]
                grad_q = 2 * errs * self.P[user_idxs, :] + self.lamb * self.Q[item_idxs, :]

                dp = np.zeros((self.n_users, self.k))
                dq = np.zeros((self.n_items, self.k))

                # aggregate the gradients
                for i in range(self.batch_size):
                    dp[user_idxs[i], :] += grad_p[i, :]
                    dq[item_idxs[i], :] += grad_q[i, :]

                # Update with momentum
                # Q_inc = self.momentum * Q_inc + self.epsilon * dq / self.batch_size
                # P_inc = self.momentum * P_inc + self.epsilon * dp / self.batch_size
                Q_inc = self.epsilon * dq
                P_inc = self.epsilon * dp

                self.P -= P_inc
                self.Q -= Q_inc

            pred = np.sum(self.P[all_user_idxs_train, :] * self.Q[all_item_idxs_train, :], axis=1)
            errs = pred - entries_train[:, 2] + self.mean_val
            obj = np.linalg.norm(errs) ** 2 + 0.5 * self.lamb * (
                    np.linalg.norm(self.P) ** 2 + np.linalg.norm(self.Q) ** 2)

            pred = np.sum(self.P[entries_val[:, 0], :] * self.Q[entries_val[:, 1], :], axis=1)
            errs = pred - entries_val[:, 2] + self.mean_val
            rmse_val = np.linalg.norm(errs) / np.sqrt(n_val)

            pred = np.sum(self.P[entries_test[:, 0], :] * self.Q[entries_test[:, 1], :], axis=1)
            errs = pred - entries_test[:, 2] + self.mean_val
            rmse_test = np.linalg.norm(errs) / np.sqrt(n_test)
            logging.info('iter {}, obj={}, rmse_val={}, rmse_test={}'.format(it, obj, rmse_val, rmse_test))
