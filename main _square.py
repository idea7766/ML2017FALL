import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import linear_model as lm
import utils

train_path = os.path.join(os.path.dirname(__file__), "./data/train.csv")
test_path = os.path.join(os.path.dirname(__file__), "./data/test.csv")
output_path = os.path.join(os.path.dirname(__file__), "./ans.csv")
model_path = os.path.join(os.path.dirname(__file__), "./model")


# 製作 "x, y" ，以每個小時預測下一個小時
x, y= utils.load(train_path, mode = 'train', fea_select = (4, 5, 6, 9))
x, max, min = utils.rescaling(x)
x_test = utils.load(test_path, mode = 'test', fea_select = (4, 5, 6, 9))


x, y = utils.shuffle(x, y)
x_train, y_train, x_val, y_val = utils.validation(x, y, ratio = 0.2)
b, w = lm.LinearRegression(x, y, lr = 0.1, epoch = 1000000, x_val = x_val, y_val = y_val)


x_test = utils.scaling(x_test, max, min) 
# print(x_test)
predicted = lm.predict(x_test, b, w)
print('>>> Predicted Result :\n', predicted)

utils.save_model(b, w, model_path)
#     x_train, y_train, x_val, y_val = utils.validation(x_temp, y_temp, ratio = 0.2)
utils.write_out_ans(predicted, output_path)