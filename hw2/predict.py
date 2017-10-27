import sys
import numpy as np

import utils
import models


X_TEST_PATH = sys.argv[1]
ANS_PATH = sys.argv[2]
MODEL_PATH = './model_sqr.npy'
SCAL_PATH = './scaler_sqr.npy'

b, w =utils.load_model(MODEL_PATH)
x_test = utils.load_data(X_TEST_PATH)
x_test = np.concatenate((x_test, x_test[:, 0:6]**2), axis=1) # 加入平方當特徵

max, min = utils.load_scaler(SCAL_PATH)
x_test = utils.scaling(x_test, max, min)

y_pred = models.predict(x_test, b, w)

utils.save_ans_dir(y_pred, ANS_PATH)
