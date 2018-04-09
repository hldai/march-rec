import numpy as np
import logging


class BiasedSVD:
    def __init__(self, n_users, n_items, k=10, learning_rate=0.1, lamb=0.1, n_epoch=20, batch_size=10):
        self.k = k
        self.lr = learning_rate
        self.lamb = lamb
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.mean_val = 0
        self.n_users = n_users
        self.n_items = n_items
        self.P = 0.1 * np.random.randn(n_users, self.k)
        self.Q = 0.1 * np.random.randn(n_items, self.k)
        self.bu = np.zeros(self.n_users, np.float32)
        self.bi = np.zeros(self.n_items, np.float32)

    def fit(self, entries_train, entries_val, entries_test):
        self.mean_val = np.mean(entries_train[:, 2])

        n_train, n_val, n_test = entries_train.shape[0], entries_val.shape[0], entries_test.shape[0]
        logging.info('PMF. {} training items, {} val items, {} test items, mean: {}.'.format(
            n_train, n_val, n_test, self.mean_val))
        logging.info('lr={}, lamb={}, batch_size={}'.format(self.lr, self.lamb, self.batch_size))

        all_users_train = entries_train[:, 0]
        all_items_train = entries_train[:, 1]
        users_val = entries_val[:, 0]
        items_val = entries_val[:, 1]
        users_test = entries_test[:, 0]
        items_test = entries_test[:, 1]

        min_rmse_val, result_obj, result_rmse_test = 1e10, 0, 0
        non_min_cnt = 0
        n_batches = int(n_train / self.batch_size)
        for it in range(self.n_epoch):
            for batch_idx in range(n_batches):
                indices = np.arange(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))

                user_idxs = np.array(entries_train[indices, 0], dtype='int32')
                item_idxs = np.array(entries_train[indices, 1], dtype='int32')

                pred = np.sum(self.P[user_idxs, :] * self.Q[item_idxs, :], axis=1
                              ) + self.mean_val + self.bu[user_idxs] + self.bi[item_idxs]
                # print(pred)
                # exit()
                errs = pred - entries_train[indices, 2]
                errs_mat = errs.reshape((-1, 1))

                # Compute gradients
                grad_p = 2 * errs_mat * self.Q[item_idxs, :] + self.lamb * self.P[user_idxs, :]
                grad_q = 2 * errs_mat * self.P[user_idxs, :] + self.lamb * self.Q[item_idxs, :]
                grad_bu = 2 * errs + self.lamb * self.bu[user_idxs]
                grad_bi = 2 * errs + self.lamb * self.bi[item_idxs]

                dp = np.zeros((self.n_users, self.k))
                dq = np.zeros((self.n_items, self.k))
                dbu = np.zeros(self.n_users, np.float32)
                dbi = np.zeros(self.n_items, np.float32)

                # aggregate the gradients
                for i in range(self.batch_size):
                    dp[user_idxs[i], :] += grad_p[i, :]
                    dq[item_idxs[i], :] += grad_q[i, :]
                    # print(dbu[user_idxs[i]], grad_bu[i])
                    dbu[user_idxs[i]] += grad_bu[i]
                    dbi[item_idxs[i]] += grad_bi[i]

                self.P -= self.lr * dp
                self.Q -= self.lr * dq
                self.bu -= self.lr * dbu
                self.bi -= self.lr * dbi

            pred = np.sum(self.P[all_users_train, :] * self.Q[all_items_train, :], axis=1
                          ) + self.mean_val + self.bu[all_users_train] + self.bi[all_items_train]
            errs = pred - entries_train[:, 2]
            obj = np.linalg.norm(errs) ** 2 + 0.5 * self.lamb * (
                    np.linalg.norm(self.P) ** 2 + np.linalg.norm(self.Q) ** 2)

            pred = np.sum(self.P[users_val, :] * self.Q[items_val, :], axis=1
                          ) + self.mean_val + self.bu[users_val] + self.bi[items_val]
            errs = pred - entries_val[:, 2]
            rmse_val = np.linalg.norm(errs) / np.sqrt(n_val)

            if rmse_val < min_rmse_val:
                min_rmse_val = rmse_val
                non_min_cnt = 0

                pred = np.sum(self.P[users_test, :] * self.Q[items_test, :], axis=1
                              ) + self.mean_val + self.bu[users_test] + self.bi[items_test]
                errs = pred - entries_test[:, 2]
                rmse_test = np.linalg.norm(errs) / np.sqrt(n_test)
                result_obj = obj
                result_rmse_test = rmse_test
                logging.info('iter {}, obj={}, rmse_val={}, rmse_test={}'.format(it, obj, rmse_val, rmse_test))
            else:
                logging.info('iter {}, obj={}, rmse_val={}'.format(it, obj, rmse_val))
                non_min_cnt += 1
                if non_min_cnt == 20:
                    break
        logging.info('END. obj={}, rmse_val={}, rmse_test={}'.format(result_obj, min_rmse_val, result_rmse_test))
