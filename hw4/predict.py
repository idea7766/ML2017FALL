import numpy as np
import sys
from gensim.models import Word2Vec as w2v
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences

import utils
import models

# set parameters
CLASS = 1
VALI = 0.1
DROP = 0.5
PATIENCE = 50
BATCH = 128

TEST_DATA = sys.argv[1]
SAVE_DIR = sys.argv[2]
MODEL = './model_bi_lstm.h5'

# load word2vec model
wv_model = w2v.load('./word_vec')
# wv_weight = wv_model.wv.syn0
## load vocab
vocab = dict([(k, v.index) for k, v in wv_model.wv.vocab.items()])

x_setence = utils.load_data(TEST_DATA, file_type = 'test')
x_setence = utils.split_setence(x_setence)
x_setence_id = utils.w2id(x_setence, vocab)

x_test = pad_sequences(x_setence_id)

model = load_model(MODEL)

predicted = model.predict(x_test)
predicted = np.around(predicted)

print(predicted)

utils.save_dir(predicted, SAVE_DIR)
