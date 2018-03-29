import numpy as np
import tensorflow as tf
import logging


class MarchRec:
    def __init__(self, n_users, n_items, k, lamb=0.1, n_epoch=20, batch_size=10, learning_rate=0.01):
        self.n_users = n_users
        self.n_items = n_items
        self.mean_val = 0
        self.k = k
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lamb = lamb
        self.lr = learning_rate
        self.hdim_bin = k
        self.n_negatives = 10

        self.P = tf.Variable(tf.random_normal([self.n_users, self.k]) * 0.1)
        self.Q = tf.Variable(tf.random_normal([self.n_items, self.k]) * 0.1)

        self.input_users = tf.placeholder(tf.int32, (None,))
        self.input_items = tf.placeholder(tf.int32, (None,))
        self.input_r_nr = tf.placeholder(tf.float32, (None,))
        P_batch = tf.gather(self.P, self.input_users)
        Q_batch= tf.gather(self.Q, self.input_items)
        self.pred_nr = tf.reduce_sum(P_batch * Q_batch, axis=1)
        self.err_sqrt_nr = tf.norm(self.input_r_nr - self.pred_nr)
        self.loss_nr = self.err_sqrt_nr ** 2 + self.lamb * (
                tf.norm(P_batch) ** 2 + tf.norm(Q_batch) ** 2)
        self.train_nr = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_nr)
        
        # self.input_users_bin = tf.placeholder(tf.int32, (None,))
        # self.input_items_bin = tf.placeholder(tf.int32, (None,))
        self.input_y_bin = tf.placeholder(tf.float32, (None,))
        # P_batch_bin = tf.gather(self.P, self.input_users_bin)
        # Q_batch_bin = tf.gather(self.Q, self.input_items_bin)
        pq_vecs = tf.concat([P_batch, Q_batch], axis=1)

        self.W_bin = tf.Variable(tf.random_uniform((2 * self.k, self.hdim_bin), -0.5, 0.5))
        self.w_out_bin = tf.Variable(tf.random_uniform((self.hdim_bin, 1), -0.5, 0.5))
        self.pred_bin = tf.matmul(tf.tanh(tf.matmul(pq_vecs, self.W_bin)), self.w_out_bin)
        self.pred_bin = tf.reshape(tf.sigmoid(self.pred_bin), (-1,))
        # self.err_bin = self.pred_bin - self.input_y_bin
        self.err_bin = tf.norm(self.pred_bin - self.input_y_bin)
        self.loss_bin = self.err_bin ** 2 + self.lamb * (tf.norm(self.W_bin) ** 2 + tf.norm(self.w_out_bin) ** 2)
        self.train_bin = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bin)

    def fit(self, entries_train, entries_val):
        r_mean = np.mean(entries_train[:, 2])
        n_train, n_val = entries_train.shape[0], entries_val.shape[0]
        logging.info('MarchRec. {} training items, {} val items, mean: {}.'.format(n_train, n_val, r_mean))
        n_batches = int(n_train / self.batch_size)

        def sep_entries(entries):
            user_idxs = np.array(entries[:, 0], dtype=np.int32)
            item_idxs = np.array(entries[:, 1], dtype=np.int32)
            r = np.array(entries[:, 2], dtype=np.int32)
            return user_idxs, item_idxs, r

        user_idxs_train, item_idxs_train, r_train = sep_entries(entries_train)
        user_idxs_val, item_idxs_val, r_val = sep_entries(entries_val)

        users_train, items_train, y_train = self.__get_train_instances(entries_train)
        users_val, items_val, y_val = self.__get_rand_val_instances(entries_val, 100)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        err_val_bin = sess.run([self.err_bin],
                               {self.input_users: users_val, self.input_items: items_val,
                                self.input_y_bin: y_val})[0]

        loss_nr = sess.run(
            [self.loss_nr],
            {self.input_users: user_idxs_train, self.input_items: item_idxs_train, self.input_r_nr: r_train})[0]
        err_sqrt_val_nr = sess.run(
            [self.err_sqrt_nr],
            {self.input_users: user_idxs_val, self.input_items: item_idxs_val, self.input_r_nr: r_val})[0]

        print(loss_nr, err_sqrt_val_nr / np.sqrt(n_val))
        print('loss_nr={}, err_val_nr={}, err_val_bin={}'.format(loss_nr, err_sqrt_val_nr, err_val_bin))
        for it in range(self.n_epoch):
            sum_loss_bin, sum_loss_nr = 0, 0
            for batch_idx in range(n_batches):
                indices = np.arange(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))

                _, loss_bin = sess.run(
                    [self.train_bin, self.loss_bin],
                    {self.input_users: users_train[indices], self.input_items: items_train[indices],
                     self.input_y_bin: y_train[indices]})
                sum_loss_bin += loss_bin / self.batch_size

                _, loss_nr = sess.run(
                    [self.train_nr, self.loss_nr],
                    {self.input_users: user_idxs_train[indices], self.input_items: item_idxs_train[indices],
                     self.input_r_nr: r_train[indices]})
                sum_loss_nr += loss_nr / self.batch_size

            err_val_nr = sess.run(
                [self.err_sqrt_nr],
                {self.input_users: user_idxs_val, self.input_items: item_idxs_val, self.input_r_nr: r_val})[0]

            err_val_bin = sess.run([self.err_bin],
                                   {self.input_users: users_val,
                                    self.input_items: items_val,
                                    self.input_y_bin: y_val})[0]
            print('loss_bin={}, loss_nr={}, err_val_nr={}, err_val_bin={}'.format(
                sum_loss_bin, sum_loss_nr, err_val_nr, err_val_bin))

    def __get_rand_val_instances(self, entries_val, n_pos_samples):
        n_neg_samples = n_pos_samples * 10
        rand_idxs = np.random.permutation(n_pos_samples)
        users_rand_val = np.zeros(n_pos_samples + n_neg_samples, np.int32)
        items_rand_val = np.zeros(n_pos_samples + n_neg_samples, np.int32)
        y_rand_val = np.zeros(n_pos_samples + n_neg_samples, np.int32)
        users_rand_val[:n_pos_samples] = entries_val[:, 0][rand_idxs]
        items_rand_val[:n_pos_samples] = entries_val[:, 1][rand_idxs]
        y_rand_val[:n_pos_samples] = 1

        pos_entries = {(user_idx, item_idx) for user_idx, item_idx, _ in entries_val}
        for i in range(n_pos_samples * 10):
            users_rand_val[n_pos_samples + i] = np.random.randint(0, self.n_users)
            items_rand_val[n_pos_samples + i] = np.random.randint(0, self.n_items)
        return users_rand_val, items_rand_val, y_rand_val

    def __get_train_instances(self, entries_train):
        pos_entries = {(user_idx, item_idx) for user_idx, item_idx, _ in entries_train}
        user_idxs, item_idxs, y_true = [], [], []
        for u, i, r in entries_train:
            user_idxs.append(u)
            item_idxs.append(i)
            y_true.append(1)
            for t in range(self.n_negatives):
                j = np.random.randint(self.n_items)
                while (u, j) in pos_entries:
                    j = np.random.randint(self.n_items)
                user_idxs.append(u)
                item_idxs.append(j)
                y_true.append(0)
        return np.asarray(user_idxs, np.int32), np.asarray(item_idxs, np.int32), np.asarray(y_true, np.int32)
