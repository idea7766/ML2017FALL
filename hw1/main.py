import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import linear_model as lm
import utils

train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

arr_train = utils.load(train_path, mode = 'train')
arr_test = utils.load(test_path, mode = 'test')
print(arr_train.shape)
print(arr_test.shape)


#製作 "x, y" ，以每個小時預測下一個小時

x, max, min = utils.rescaling(arr_train[:-1])
y = arr_train[1:,9]
print(x.shape)
print(y.shape)

# b, w = lm.LinearRegression(x, y, lr = 0.01, epoch = 100000)
b, w = lm.LinearRegression_close(x, y)

x_test = utils.scaling(arr_test, max, min) 
predicted = lm.predcit(x_test, b, w)

utils.write_out_ans(predicted, output_path)
print(predicted)
'''
fig, ax =plt.subplots()
#點座標
ax.scatter(y, predicted)
#劃線
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
#欄位名稱
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''