import sys
import numpy as np

import utils
import models

# X_TRAIN_PATH = './data/X_train'
# Y_TRAIN_PATH = './data/Y_train'
# X_TEST_PATH = './data/X_test'
# ANS_PATH = sys.argv[4]

SCALER_PATH = './scaler'
MODEL_PATH = './model/model'

X_TRAIN_PATH = sys.argv[1]
Y_TRAIN_PATH = sys.argv[2]
X_TEST_PATH = sys.argv[3]
ANS_PATH = sys.argv[4]


x = utils.load_data(X_TRAIN_PATH)
y = utils.load_data(Y_TRAIN_PATH).flatten()
x_test = utils.load_data(X_TEST_PATH)

x, max, min = utils.rescaling(x)
x_test = utils.scaling(x_test, max, min)

b, w = models.logistic_regression(x, y, lr = 1, epoch = 10000, validation_rate = 0.1, optimizer = 'adagrad', early_stopping = True, patience = 10)

y_pred = models.predict(x_test, b, w)

# print(y_pred)

# utils.save_ans(y_pred)
utils.save_ans_dir(y_pred, ANS_PATH)
# utils.save_model(b, w, MODEL_PATH)
# utils.save_scaler(max, min, SCALER_PATH)