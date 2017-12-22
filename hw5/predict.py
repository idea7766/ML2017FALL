import numpy as np
import sys
from keras.models import load_model

import utils
import models

MEAN = 3.58171208604

TEST_DATA = sys.argv[1]
ANS_PATH = sys.argv[2]
MODEL = './model.h5'

x_test = utils.load_data(TEST_DATA, file_type = 'test')
model = load_model(MODEL, custom_objects = {'rmse':models.rmse})

y_pred = model.predict(np.hsplit(x_test, 2))
print('--y_pred:\n', y_pred)
# y_pred = utils.post_process(y_pred, bias = MEAN)
y_pred = utils.post_process(y_pred, round = False, bias = MEAN)
print('--ans:\n', y_pred)

utils.save_dir(data = y_pred, path = ANS_PATH)