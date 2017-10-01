import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import linear_model as lm
import utils

data_path = sys.argv[1]
# output_path = sys.argv[2]

df_pm25 = utils.load(data_path)

#將 'NR' 改成 0
arr_pm25_pre = df_pm25.iloc[:,3:].replace('NR', 0).values
arr_pm25_pre = arr_pm25_pre.astype(float)

arr_pm25 = np.array([arr_pm25_pre[:18, 0]])
for i in range(240):
    for j in range(24):
        if i==0 and j == 0:
            continue
        arr_pm25 = np.append(arr_pm25, [arr_pm25_pre[18*i:18+18*i, j]], axis = 0)

#製作 "x, y" ，以每個小時預測下一個小時
x = arr_pm25[:-1, :]
y = arr_pm25[1:,9]
x, max, min = utils.rescaling(arr_pm25)
print(x.shape)
print(y.shape)
b, w = lm.LinearRegression(x[:-1], y, lr = 0.01, epoch = 100000)
predicted = lm.predcit(x[:-1], b, w)

# print('coef:', lr.coef_)
# print('mean square:', mean_squared_error(y, predicted))

fig, ax =plt.subplots()
#點座標
ax.scatter(y, predicted)
#劃線
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
#欄位名稱
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
