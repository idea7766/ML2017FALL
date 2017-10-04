import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import linear_model as lm
import utils

train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

arr_train = utils.load(train_path, mode = 'train', fea_select = (4, 5, 6, 9))
arr_test = utils.load(test_path, mode = 'test', fea_select = (4, 5, 6, 9))

# np.random.shuffle(arr_train)


#製作 "x, y" ，以每個小時預測下一個小時
x, max, min = utils.rescaling(arr_train[:-1])
y = arr_train[1:,3]

print(x.shape)
print(y.shape)

b, w = np.array([0]), np.zeros(x.shape[1])
for i in range(5):
    x_temp, y_temp = utils.shuffle(x, y)
    # print(x_temp.shape)
    # print(y_temp.shape)
    x_train, y_train, x_val, y_val = utils.validation(x_temp, y_temp, ratio = 0.2)
    # print(x_train.shape)
    # print(y_train.shape)
    b_temp, w_temp = lm.LinearRegression_close(x_train, y_train)
    b = b + b_temp
    w = w + w_temp
b = b / 5
w = w / 5    
print(b)
print(w.shape)
# b, w = lm.LinearRegression(x, y, lr = 0.01, epoch = 100000)
# b, w = lm.LinearRegression_close(x, y)

x_test = utils.scaling(arr_test, max, min) 
print(x_test)
predicted = lm.predict(x_test, b, w)
print(predicted)

utils.write_out_ans(predicted, output_path)