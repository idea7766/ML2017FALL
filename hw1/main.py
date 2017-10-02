import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import linear_model as lm
import utils

data_path = sys.argv[1]
data_path = sys.argv[2]
# output_path = sys.argv[2]

arr_train = utils.load(data_path, mode = 'train')


#製作 "x, y" ，以每個小時預測下一個小時
x = arr_train[:-1, :]
y = arr_train[1:,9]
x, max, min = utils.rescaling(arr_train)
print(x.shape)
print(y.shape)

b, w = lm.LinearRegression(x[:-1], y, lr = 0.01, epoch = 100000)
predicted = lm.predcit(x[:-1], b, w)


fig, ax =plt.subplots()
#點座標
ax.scatter(y, predicted)
#劃線
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
#欄位名稱
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()