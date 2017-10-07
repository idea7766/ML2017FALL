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

fea_select, y_pos = (0, 4, 5, 6, 7, 8, 9, 16), 70

# 製作 "x, y" ，以每個小時預測下一個小時
x, y= utils.load(train_path, mode = 'train', fea_select = fea_select, y_pos = y_pos)
x = np.concatenate((x,x**2), axis=1)
x, max, min = utils.rescaling(x)

x_test = utils.load(test_path, mode = 'test', fea_select = fea_select, y_pos = y_pos)
x_test = np.concatenate((x_test,x_test**2), axis=1)


# 做 5 回
# b, w = np.array([0]), np.zeros(x.shape[1])
# for i in range(5):
#     x_temp, y_temp = utils.shuffle(x, y)
#     print('x temp: ',x_temp.shape)
#     print('y temp', y_temp.shape)
#     x_train, y_train, x_val, y_val = utils.validation(x_temp, y_temp, ratio = 0.2)
#     print('x train: ',x_train.shape)
#     print('y train', y_train.shape)
#     # b_temp, w_temp = lm.LinearRegression_close(x_train, y_train)
#     b_temp, w_temp = lm.LinearRegression(x_train, y_train, lr = 0.01, epoch = 10000, x_val = x_val, y_val = y_val)
#     b = b + b_temp
#     w = w + w_temp
# b = b / 5
# w = w / 5    

x, y = utils.shuffle(x, y)
x_train, y_train, x_val, y_val = utils.validation(x, y, ratio = 0.1)
b, w = lm.LinearRegression(x, y, lr = 100000, epoch = 1000000, lr_method = 'adagrad', x_val = x_val, y_val = y_val)
# b, w = lm.LinearRegression_close(x, y)


x_test = utils.scaling(x_test, max, min) 
# print(x_test)
predicted = lm.predict(x_test, b, w)
print('>>> Predicted Result :\n', predicted)

utils.save_model(b, w, model_path)
#     x_train, y_train, x_val, y_val = utils.validation(x_temp, y_temp, ratio = 0.2)
utils.write_out_ans(predicted, output_path)
