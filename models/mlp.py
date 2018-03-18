import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras import layers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dense, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import evaluate


class MLPRec:
    def __init__(self, n_users, n_items, layer_dims, learning_rate=0.1, lamb=0.01, n_epochs=10):
        self.n_users = n_users
        self.n_items = n_items
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.n_negatives = 10
        self.batch_size = 10

        n_layers = len(layer_dims)

        # Input variables
        input_user = Input(shape=(1,), dtype='int32', name='user_input')
        input_item = Input(shape=(1,), dtype='int32', name='item_input')

        initer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        embeddings_user = Embedding(input_dim=n_users, output_dim=int(layer_dims[0] / 2), name='user_embedding',
                                    embeddings_initializer=initer, input_length=1)
        embeddings_item = Embedding(input_dim=n_items, output_dim=int(layer_dims[0] / 2), name='item_embedding',
                                    embeddings_initializer=initer, input_length=1)

        user_latent = Flatten()(embeddings_user(input_user))
        item_latent = Flatten()(embeddings_item(input_item))

        vec = layers.concatenate([user_latent, item_latent])
        # MLP layers
        for i in range(1, n_layers):
            layer = Dense(layer_dims[i], kernel_regularizer=l2(lamb), activation='relu', name='layer%d' % i)
            vec = layer(vec)

        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vec)

        self.model = Model(inputs=[input_user, input_item], outputs=[prediction])
        self.model.compile(Adagrad(lr=learning_rate), loss='mean_squared_error')

    def fit(self, entries_train, entries_val):
        pos_entries_train = {(user_idx, item_idx) for user_idx, item_idx, _ in entries_train}

        n_val_samples = 100
        rand_idxs = np.random.permutation(n_val_samples)
        # rand_val_entries = entries_val[rand_idxs]
        rand_val_entries = entries_train[rand_idxs]
        neg_users, neg_items = np.zeros(n_val_samples * 10, np.int32), np.zeros(n_val_samples * 10, np.int32)
        for i in range(n_val_samples * 10):
            neg_users[i] = np.random.randint(0, self.n_users)
            neg_items[i] = np.random.randint(0, self.n_items)

        y_pred_pos = self.model.predict([rand_val_entries[:, 0], rand_val_entries[:, 1]]).reshape((-1))
        y_pred_neg = self.model.predict([neg_users, neg_items]).reshape((-1))
        err = np.sum((1 - y_pred_pos) ** 2) + np.sum(y_pred_neg ** 2)
        print(np.sqrt(err) / np.sqrt(n_val_samples * 2))

        for it in range(self.n_epochs):
            user_idxs, item_idxs, y_true = self.__get_train_instances(entries_train, pos_entries_train)
            print('training instances generated.')
            hist = self.model.fit(
                [user_idxs, item_idxs], y_true, batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
            print('loss={}'.format(hist.history['loss'][0]))

            y_pred_pos = self.model.predict([rand_val_entries[:, 0], rand_val_entries[:, 1]]).reshape((-1))
            y_pred_neg = self.model.predict([neg_users, neg_items]).reshape((-1))
            err = np.sum((1 - y_pred_pos) ** 2) + np.sum(y_pred_neg ** 2)
            print(np.sqrt(err) / np.sqrt(n_val_samples * 2))

    def __get_train_instances(self, entries_train, pos_entries):
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
