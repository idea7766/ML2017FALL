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


# arr_train = utils.load(train_path, mode = 'train', fea_select = (4, 5, 6, 9))
x, y= utils.load(train_path, mode = 'train', fea_select = (4, 5, 6, 9))
print(x)
print(y)


arr_test = utils.load(test_path, mode = 'test', fea_select = (4, 5, 6, 9))

# np.random.shuffle(arr_train)
print()
# x, max, min = utils.rescaling(x)

#製作 "x, y" ，以每個小時預測下一個小時
# x, max, min = utils.rescaling(arr_train[:-1])
# x = arr_train[:-1, :]
# y = arr_train[1:,-1]

b, w = np.array([0]), np.zeros(x.shape[1])
for i in range(5):
    x_temp, y_temp = utils.shuffle(x, y)
    print('x temp: ',x_temp.shape)
    print('y temp', y_temp.shape)
    x_train, y_train, x_val, y_val = utils.validation(x_temp, y_temp, ratio = 0.2)
    print('x train: ',x_train.shape)
    print('y train', y_train.shape)
    # b_temp, w_temp = lm.LinearRegression_close(x_train, y_train)
    b_temp, w_temp = lm.LinearRegression(x_train, y_train, lr = 0.01, epoch = 10000, x_val = x_val, y_val = y_val)
    b = b + b_temp
    w = w + w_temp
b = b / 5
w = w / 5    
print('b: ',b)
print('w shape: ', w.shape)
# b, w = lm.LinearRegression(x, y, lr = 0.01, epoch = 100000)
# b, w = lm.LinearRegression_close(x, y)

x_test = arr_test
# x_test = utils.scaling(arr_test, max, min) 
# print(x_test)
predicted = lm.predict(x_test, b, w)
print(predicted)

utils.write_out_ans(predicted, output_path)