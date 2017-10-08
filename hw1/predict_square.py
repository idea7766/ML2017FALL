import sys
import os
import pandas as pd
import numpy as np

import utils
import linear_model as lm

test_path = sys.arg[1]
output_path = sys.arg[2]
# test_path = os.path.join(os.path.dirname(__file__), "./data/test.csv")
# output_path = os.path.join(os.path.dirname(__file__), "./ans_sqr_test.csv")

model_path = os.path.join(os.path.dirname(__file__), "./model_sqr.npy")
scaler_path = os.path.join(os.path.dirname(__file__), './scaler_sqr.npy')

fea_select, y_pos = (0, 4, 5, 6, 7, 8, 9, 16), 70

b, w = utils.load_model(model_path)
# print(w.shape)
max, min = utils.load_scaler(scaler_path)
x_test = utils.load(test_path, mode = 'test', fea_select = fea_select, y_pos = y_pos)
x_test = np.concatenate((x_test, x_test ** 2), axis = 1)
x_test = utils.scaling(x_test, max, min)

predicted = lm.predict(x_test, b, w)
print('Predicted:', predicted)

utils.save_ans(predicted, output_path)