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

        self.P = tf.Variable(tf.random_normal([self.n_users, self.k]) * 0.1)
        self.Q = tf.Variable(tf.random_normal([self.n_items, self.k]) * 0.1)

        # self.input_users = tf.placeholder(tf.int32, (None,))
        # self.input_items = tf.placeholder(tf.int32, (None,))
        # self.input_r = tf.placeholder(tf.float32, (None,))
        # P_batch = tf.gather(self.P, self.input_users)
        # Q_batch = tf.gather(self.Q, self.input_items)
        # self.r_pred = tf.reduce_sum(P_batch * Q_batch, axis=1)
        # self.err_sqrt = tf.norm(self.input_r - self.r_pred)
        # self.loss = self.err_sqrt ** 2 + self.lamb * (tf.norm(P_batch) ** 2 + tf.norm(Q_batch) ** 2)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # self.train = optimizer.minimize(self.loss)
        
        self.input_users_bin = tf.placeholder(tf.int32, (None,))
        self.input_items_bin = tf.placeholder(tf.int32, (None,))
        self.input_y_bin = tf.placeholder(tf.int32, (None,))
        P_batch_bin = tf.gather(self.P, self.input_users_bin)
        Q_batch_bin = tf.gather(self.Q, self.input_items_bin)
        pq_vecs = tf.concat([P_batch_bin, Q_batch_bin], axis=1)

        self.W_bin = tf.Variable(tf.random_uniform((2 * self.k, self.hdim_bin), -0.5, 0.5))
        self.w_out_bin = tf.Variable(tf.random_uniform((self.hdim_bin, 1), -0.5, 0.5))
        self.pred_bin = tf.matmul(tf.tanh(tf.matmul(pq_vecs, self.W_bin)), self.w_out_bin)
        self.pred_bin = tf.sigmoid(self.pred_bin)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        v = sess.run([self.pred_bin], {self.input_users_bin: [0, 1],
                                       self.input_items_bin: [2, 3]})
        print(v[0])
        exit()

    def fit(self, entries_train, entries_val):
        self.mean_val = np.mean(entries_train[:, 2])

        n_train, n_val = entries_train.shape[0], entries_val.shape[0]
        logging.info('MarchRec. {} training items, {} val items, mean: {}.'.format(n_train, n_val, self.mean_val))

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        n_batches = int(n_train / self.batch_size)

        def sep_entries(entries):
            user_idxs = np.array(entries[:, 0], dtype=np.int32)
            item_idxs = np.array(entries[:, 1], dtype=np.int32)
            r = np.array(entries[:, 2], dtype=np.int32)
            return user_idxs, item_idxs, r

        user_idxs_train, item_idxs_train, r_train = sep_entries(entries_train)
        user_idxs_val, item_idxs_val, r_val = sep_entries(entries_val)

        loss = sess.run(
            [self.loss],
            {self.input_users: user_idxs_train, self.input_items: item_idxs_train, self.input_r: r_train})[0]
        err_sqrt_val = sess.run(
            [self.err_sqrt],
            {self.input_users: user_idxs_val, self.input_items: item_idxs_val, self.input_r: r_val})[0]
        print(loss, err_sqrt_val / np.sqrt(n_val))

        for it in range(self.n_epoch):
            losses = list()
            for batch_idx in range(n_batches):
                indices = np.arange(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))

                _, loss = sess.run(
                    [self.train, self.loss],
                    {self.input_users: user_idxs_train[indices], self.input_items: item_idxs_train[indices],
                     self.input_r: r_train[indices]})
                losses.append(loss)

            err_sqrt_val = sess.run(
                [self.err_sqrt],
                {self.input_users: user_idxs_val, self.input_items: item_idxs_val, self.input_r: r_val})[0]

            print(sum(losses), err_sqrt_val / np.sqrt(n_val))
