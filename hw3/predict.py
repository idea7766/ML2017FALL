import os
import sys
from keras.models import Sequential, load_model

import utils
import read_n_write as rw

base_dir = os.path.dirname(__file__)

MODEL_PATH = './model_re_bst.h5'
TEST_PATH = sys.argv[1]
SAVE_PATH = sys.argv[2]

# io processing
model  = load_model(MODEL_PATH)
x = rw.read_dataset(TEST_PATH, labeled_data = False)

y_pred = model.predict(x)
y_pred = utils.to_nb_class(y_pred)
rw.save_dir(y_pred, path = SAVE_PATH)