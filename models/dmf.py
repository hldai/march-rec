import tensorflow as tf
import numpy as np
import sys
import os
import heapq
import math


class DMF:
    def __init__(self, n_users, n_items, entries_train, entries_val, entries_test, num_neg=10, user_layer=(512, 64),
                 item_layer=(1024, 64), learning_rate=0.0001, n_epochs=50, batch_size=256, early_stop=5, top_k=10,
                 checkpoint_dir='checkpoints'):
        self.n_users = n_users
        self.n_items = n_items
        self.data_name = 'ml-1m'
        self.num_neg = num_neg
        self.lr = learning_rate
        self.user_layer = user_layer
        self.item_layer = item_layer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.top_k = top_k
        self.cp_dir = checkpoint_dir

        self.x_ui = tf.convert_to_tensor(self.__get_ui_matrix(entries_train))
        self.x_iu = tf.transpose(self.x_ui)

        self.users = tf.placeholder(tf.int32)
        self.items = tf.placeholder(tf.int32)
        self.rates = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

        input_users = tf.nn.embedding_lookup(self.x_ui, self.users)
        input_items = tf.nn.embedding_lookup(self.x_iu, self.items)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        user_layer_dims = [self.n_items] + self.user_layer
        output_u = input_users
        with tf.name_scope("User_Layer"):
            for i in range(0, len(user_layer_dims) - 1):
                W = init_variable([user_layer_dims[i], user_layer_dims[i + 1]], "user_W" + str(i + 2))
                b = init_variable([user_layer_dims[i + 1]], "user_b" + str(i + 2))
                output_u = tf.nn.relu(tf.add(tf.matmul(output_u, W), b))

            # W1_user = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            # user_out = tf.matmul(input_users, user_W1)
            # for i in range(0, len(self.userLayer)-1):
            #     W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
            #     b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
            #     user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        item_layer_dims = [self.n_users] + self.item_layer
        output_i = input_items
        with tf.name_scope("Item_Layer"):
            for i in range(0, len(item_layer_dims) - 1):
                W = init_variable([item_layer_dims[i], item_layer_dims[i + 1]], "item_W" + str(i + 2))
                b = init_variable([item_layer_dims[i + 1]], "item_b" + str(i + 2))
                output_i = tf.nn.relu(tf.add(tf.matmul(output_i, W), b))

            # item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            # item_out = tf.matmul(input_items, item_W1)
            # for i in range(0, len(self.itemLayer)-1):
            #     W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
            #     b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
            #     item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (
                norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

        regRate = self.rate / self.maxRate
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        self.loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def __get_ui_matrix(self, entries):
        x = np.zeros((self.n_users, self.n_items), np.float32)
        for u, i, r in entries:
            x[u][i] = r
        return x

    def fit(self, entries_train, entries_val, entries_test):
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("Start Evaluation!")
            hr, NDCG = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)
