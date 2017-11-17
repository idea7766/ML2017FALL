import os
from keras.models import Sequential, load_model

import utils
import read_n_write as rw

base_dir = os.path.dirname(__file__)
MODEL_PATH = './model_bst.h5'

# io processing
model  = load_model(MODEL_PATH)
x = rw.read_dataset('data/test.csv', labeled_data = False)
x_train, y_train, _ = rw.read_dataset('data/train.csv', shuffle = True)

new_model = utils.self_learn(model, x, x_train, y_train,7, 0.5)
rw.save_model(new_model, 'model_re')

result = new_model.predict_classes(x)
rw.save(result)