import sys
import os
import pandas as pd
import numpy as np

import utils
import linear_model as lm

test_path = sys.arg[1]
output_path = sys.arg[2]
# test_path = os.path.join(os.path.dirname(__file__), "./data/test.csv")
# output_path = os.path.join(os.path.dirname(__file__), "./ans_test.csv")

model_path = os.path.join(os.path.dirname(__file__), "./model.npy")
scaler_path = os.path.join(os.path.dirname(__file__), './scaler.npy')

fea_select, y_pos = (0, 4, 5, 6, 7, 8, 9, 16), 70

b, w = utils.load_model(model_path)
max, min = utils.load_scaler(scaler_path)
test_x = utils.load(test_path, mode = 'test', fea_select = fea_select, y_pos = y_pos)
test_x = utils.scaling(test_x, max, min)

predicted = lm.predict(test_x, b, w)
print('Predicted:', predicted)

utils.save_ans(predicted, output_path)