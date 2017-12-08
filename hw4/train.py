import os
import numpy as np
import sys
from gensim.models import Word2Vec as w2v
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

import utils
import models

# set parameters
CLASS = 1
VALI = 0.1
DROP = 0.5
PATIENCE = 4
BATCH = 128

TRAIN_DATA = sys.argv[1]
WORD_VEC = './word_vec'

TRAIN_FOLDER = './model_chk/bi_lstm_log_0'
CHECK_POINT_NAME = TRAIN_FOLDER + '/model.{epoch:04d}-{val_acc:.4f}.h5'
LOG_NAME = TRAIN_FOLDER + '/log.csv'

if not os.path.exists(TRAIN_FOLDER):
    os.makedirs(TRAIN_FOLDER)

# load word2vec model
wv_model = w2v.load(WORD_VEC)
# wv_weight = wv_model.wv.syn0
## load vocab
vocab = dict([(k, v.index) for k, v in wv_model.wv.vocab.items()])

y_train, x_setence = utils.load_data(TRAIN_DATA, file_type = 'train')
x_setence = utils.split_setence(x_setence)
x_setence_id = utils.w2id(x_setence, vocab)

x_train = pad_sequences(x_setence_id)

# add embedding matrix
embedding_matrix = utils.get_embedding_matrix(wv_model)


# build keraa model 
model = models.build_LSTM(num_word = len(vocab), num_class = CLASS,
                         embedding_matrix = [embedding_matrix],
                         drop_rate = DROP)

# callback
callback = [
            TensorBoard(log_dir=TRAIN_FOLDER),
            CSVLogger(LOG_NAME, append=True),
            ModelCheckpoint(CHECK_POINT_NAME, period=1),
            ReduceLROnPlateau('val_loss', factor=0.1, patience=int(PATIENCE/4), verbose=1),
            EarlyStopping(patience = PATIENCE)
            ]

model.fit(x_train, y_train, batch_size = BATCH, epochs = 100, callbacks=callback, validation_split=0.05)
# model.fit_generator(utils.vec_setence_gen(x_setence, y.reshape(-1, 1),model = wv_model),
#                     samples_per_epoch = len(x_setence),
#                     epochs = 1, callbacks = callback)
