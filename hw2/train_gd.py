import sys
import numpy as np

import utils
import models

# X_TRAIN_PATH = './data/X_train'
# Y_TRAIN_PATH = './data/Y_train'
# X_TEST_PATH = './data/X_test'
SCALER_PATH = './scaler'
MODEL_PATH = './model/model_gd'

X_TRAIN_PATH = sys.argv[1]
Y_TRAIN_PATH = sys.argv[2]
X_TEST_PATH = sys.argv[3]
ANS_PATH = sys.argv[4]

x = utils.load_data(X_TRAIN_PATH)
y = utils.load_data(Y_TRAIN_PATH).flatten()
x_test = utils.load_data(X_TEST_PATH)

x, max, min = utils.rescaling(x)
x_test = utils.scaling(x_test, max, min)

b, w = models.gaussian_distribution(x, y)

y_pred = models.predict(x_test, b, w)

# print(y_pred)

# train_acc = models.acc(y, models.predict(x, b, w))
# print(train_acc)

utils.save_ans_dir(y_pred, ANS_PATH)
# utils.save_ans(y_pred, 'ans_gd')
# utils.save_model(b, w, MODEL_PATH)
# utils.save_scaler(max, min, SCALER_PATH)