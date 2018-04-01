import numpy as np
import tensorflow as tf
import logging


class MarchRec:
    def __init__(self, n_users, n_items, k, n_epoch=20, batch_size=10, learning_rate=0.01,
                 alpha1=1, alpha2=1, alpha3=1, lamb1=0.1, lamb2=0.1, lamb3=0.1):
        self.n_users = n_users
        self.n_items = n_items
        self.mean_val = 0
        self.k = k
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3
        self.lr = learning_rate
        self.hdim_bin = k
        self.hdim_pr = k
        self.n_negatives = 10
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.P = tf.Variable(tf.random_normal([self.n_users, self.k]) * 0.1)
        self.Q = tf.Variable(tf.random_normal([self.n_items, self.k]) * 0.1)

        self.input_users = tf.placeholder(tf.int32, (None,))
        self.input_items = tf.placeholder(tf.int32, (None,))
        self.input_r = tf.placeholder(tf.float32, (None,))
        P_batch = tf.gather(self.P, self.input_users)
        Q_batch= tf.gather(self.Q, self.input_items)
        self.pred_nr = tf.reduce_sum(P_batch * Q_batch, axis=1)
        self.err_nr = tf.norm(self.input_r - self.pred_nr)
        # self.loss_nr = self.err_nr ** 2 + self.lamb1 * (
        #         tf.norm(P_batch) ** 2 + tf.norm(Q_batch) ** 2)
        self.loss_nr = self.err_nr ** 2
        self.loss_nr_train = self.err_nr ** 2 + self.lamb1 * (
                tf.norm(P_batch) ** 2 + tf.norm(Q_batch) ** 2)
        # self.train_nr = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_nr)
        self.train_nr = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss_nr_train)
        
        self.input_users_bin = tf.placeholder(tf.int32, (None,))
        self.input_items_bin = tf.placeholder(tf.int32, (None,))
        self.input_y_bin = tf.placeholder(tf.float32, (None,))
        P_batch_bin = tf.gather(self.P, self.input_users_bin)
        Q_batch_bin = tf.gather(self.Q, self.input_items_bin)
        pq_vecs = tf.concat([P_batch_bin, Q_batch_bin], axis=1)
        # pq_vecs = tf.concat([P_batch, Q_batch], axis=1)

        self.W_bin = tf.Variable(tf.random_uniform((2 * self.k, self.hdim_bin), -0.5, 0.5))
        self.w_out_bin = tf.Variable(tf.random_uniform((self.hdim_bin, 1), -0.5, 0.5))
        self.pred_bin = tf.matmul(tf.tanh(tf.matmul(pq_vecs, self.W_bin)), self.w_out_bin)
        self.pred_bin = tf.reshape(tf.sigmoid(self.pred_bin), (-1,))
        # self.err_bin = self.pred_bin - self.input_y_bin
        self.err_bin = tf.norm(self.pred_bin - self.input_y_bin)
        self.loss_bin = self.err_bin ** 2
        self.loss_bin_train = self.err_bin ** 2 + self.lamb1 * (
                tf.norm(P_batch_bin) ** 2 + tf.norm(Q_batch_bin) ** 2) + self.lamb2 * (
                tf.norm(self.W_bin) ** 2 + tf.norm(self.w_out_bin) ** 2)
        # self.train_bin = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bin)
        self.train_bin = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss_bin_train)

        self.Wu = tf.Variable(tf.random_uniform((self.k, self.hdim_pr), -0.5, 0.5))
        self.Wi = tf.Variable(tf.random_uniform((self.k, self.hdim_pr), -0.5, 0.5))
        self.vu = tf.tanh(tf.matmul(P_batch, self.Wu))
        self.vi = tf.tanh(tf.matmul(Q_batch, self.Wi))
        self.pred_pr = tf.reduce_sum(self.vu * self.vi, axis=1)
        self.err_pr = tf.norm(self.pred_pr - self.input_r)
        self.loss_pr = self.err_pr ** 2
        self.loss_pr_train = self.err_pr ** 2 + self.lamb1 * (
                tf.norm(P_batch) ** 2 + tf.norm(Q_batch) ** 2) + self.lamb3 * (
                tf.norm(self.Wu) ** 2 + tf.norm(self.Wi) ** 2)
        # self.train_pr = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_pr)
        self.train_pr = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss_pr)

        self.loss = alpha1 * self.loss_nr + alpha2 * self.loss_bin + self.loss_pr + self.lamb1 * (
                tf.norm(P_batch) ** 2 + tf.norm(Q_batch) ** 2) + self.lamb1 * (
                tf.norm(P_batch_bin) ** 2 + tf.norm(Q_batch_bin) ** 2) + self.lamb2 * (
                tf.norm(self.W_bin) ** 2 + tf.norm(self.w_out_bin) ** 2) + self.lamb3 * (
                tf.norm(self.Wu) ** 2 + tf.norm(self.Wi) ** 2)
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def fit(self, entries_train, entries_val, entries_test):
        r_mean = np.mean(entries_train[:, 2])
        n_train, n_val, n_test = entries_train.shape[0], entries_val.shape[0], entries_test.shape[0]
        logging.info('k={}, lr={}, lamb1={}, lamb2={}, lamb3={}, alpha1={}, alpha2={}, alpha2={}'.format(
            self.k, self.lr, self.lamb1, self.lamb2, self.lamb3, self.alpha1, self.alpha2, self.alpha3))
        logging.info('MarchRec. {} training items, {} val items, mean: {}.'.format(n_train, n_val, r_mean))
        n_batches = int(n_train / self.batch_size)

        def sep_entries(entries):
            user_idxs = np.array(entries[:, 0], dtype=np.int32)
            item_idxs = np.array(entries[:, 1], dtype=np.int32)
            r = np.array(entries[:, 2], dtype=np.int32)
            return user_idxs, item_idxs, r

        users_train_r, items_train_r, r_train = sep_entries(entries_train)
        users_val_r, items_val_r, r_val = sep_entries(entries_val)
        users_test_r, items_test_r, r_test = sep_entries(entries_test)

        users_train_bin, items_train_bin, y_train = self.__get_train_instances(entries_train)
        users_val_bin, items_val_bin, y_val = self.__get_rand_val_instances(entries_val, 100)

        batch_size_bin = int(users_train_bin.shape[0] / n_batches)
        print(n_train, users_train_bin.shape[0], batch_size_bin, self.batch_size)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        err_val_bin = sess.run([self.err_bin],
                               {self.input_users_bin: users_val_bin, self.input_items_bin: items_val_bin,
                                self.input_y_bin: y_val})[0]
        err_val_bin /= np.sqrt(n_val)

        loss_nr = sess.run(
            [self.loss_nr],
            {self.input_users: users_train_r, self.input_items: items_train_r, self.input_r: r_train})[0]
        err_val_pr, err_val_nr = sess.run(
            [self.err_pr, self.err_nr],
            {self.input_users: users_val_r, self.input_items: items_val_r, self.input_r: r_val})
        err_val_nr /= np.sqrt(n_val)
        err_val_pr /= np.sqrt(n_val)
        err_test_pr = sess.run(
            [self.err_pr],
            {self.input_users: users_test_r, self.input_items: items_test_r, self.input_r: r_test})[0]
        err_test_pr /= np.sqrt(n_test)

        logging.info(
            'loss_nr={:.3f}, err_val_nr={:.5f}, err_val_bin={:.5f}, err_val_pr={:.5f}, err_test_pr={:.5f}'.format(
                loss_nr, err_val_nr, err_val_bin, err_val_pr, err_test_pr))
        min_err_val_pr, inf_cnt = 1e10, 0
        for it in range(self.n_epoch):
            sum_loss_bin, sum_loss_nr, sum_loss_pr = 0, 0, 0
            for batch_idx in range(n_batches):
                indices = np.arange(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))
                indices_bin = np.arange(batch_size_bin * batch_idx, batch_size_bin * (batch_idx + 1))

                _, loss, loss_pr, loss_nr, loss_bin = sess.run(
                    [self.train, self.loss, self.loss_pr, self.loss_nr, self.loss_bin],
                    {self.input_users: users_train_r[indices], self.input_items: items_train_r[indices],
                     self.input_users_bin: users_train_bin[indices_bin],
                     self.input_items_bin: items_train_bin[indices_bin],
                     self.input_r: r_train[indices], self.input_y_bin: y_train[indices_bin]}
                )

                # _, loss_bin = sess.run(
                #     [self.train_bin, self.loss_bin],
                #     {self.input_users: users_train_bin[indices_bin], self.input_items: items_train_bin[indices_bin],
                #      self.input_y_bin: y_train[indices_bin]})
                #
                # _, loss_nr = sess.run(
                #     [self.train_nr, self.loss_nr],
                #     {self.input_users: users_train_r[indices], self.input_items: items_train_r[indices],
                #      self.input_r: r_train[indices]})
                #
                # _, loss_pr = sess.run(
                #     [self.train_pr, self.loss_pr],
                #     {self.input_users: users_train_r[indices], self.input_items: items_train_r[indices],
                #      self.input_r: r_train[indices]})
                sum_loss_pr += loss_pr / self.batch_size
                sum_loss_nr += loss_nr / self.batch_size
                sum_loss_bin += loss_bin / batch_size_bin

            err_val_pr, err_val_nr = sess.run(
                [self.err_pr, self.err_nr],
                {self.input_users: users_val_r, self.input_items: items_val_r, self.input_r: r_val})
            err_val_nr /= np.sqrt(n_val)
            err_val_pr /= np.sqrt(n_val)

            err_val_bin = sess.run([self.err_bin],
                                   {self.input_users_bin: users_val_bin,
                                    self.input_items_bin: items_val_bin,
                                    self.input_y_bin: y_val})[0]
            err_val_bin /= np.sqrt(n_val)

            if err_val_pr < min_err_val_pr:
                min_err_val_pr = err_val_pr
                inf_cnt = 0
            else:
                inf_cnt += 1
            if inf_cnt == 20:
                break

            err_test_pr = sess.run(
                [self.err_pr],
                {self.input_users: users_test_r, self.input_items: items_test_r, self.input_r: r_test})[0]
            err_test_pr /= np.sqrt(n_test)

            logging.info('it={}, l_pr={:.3f}, l_bin={:.3f}, l_nr={:.3f}, err_val_pr={:.5f}, err_test_pr={:.5f}, '
                         'err_val_nr={:.4f}, err_val_bin={:.4f}'.format(
                it, sum_loss_pr, sum_loss_bin, sum_loss_nr, err_val_pr, err_test_pr, err_val_nr, err_val_bin))

    def __get_rand_val_instances(self, entries_val, n_pos_samples):
        n_neg_samples = n_pos_samples * 10
        rand_idxs = np.random.permutation(n_pos_samples)
        users_rand_val = np.zeros(n_pos_samples + n_neg_samples, np.int32)
        items_rand_val = np.zeros(n_pos_samples + n_neg_samples, np.int32)
        y_rand_val = np.zeros(n_pos_samples + n_neg_samples, np.int32)
        users_rand_val[:n_pos_samples] = entries_val[:, 0][rand_idxs]
        items_rand_val[:n_pos_samples] = entries_val[:, 1][rand_idxs]
        y_rand_val[:n_pos_samples] = 1

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
