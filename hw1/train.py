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
scaler_path = os.path.join(os.path.dirname(__file__), "./scaler")
 
fea_select, y_pos = (0, 4, 5, 6, 7, 8, 9, 16), 70


x, y= utils.load(train_path, mode = 'train', fea_select = fea_select, y_pos = y_pos) # 讀出所有 data 、擷取 feature 、劃分 9 天成一筆
x, max, min = utils.rescaling(x) # 作 rescaling ， 在 [0, 1] 間
x, y = utils.shuffle(x, y)

x_train, y_train, x_val, y_val = utils.validation(x, y, ratio = 0.1)
b, w = lm.LinearRegression(x, y, lr = 100000, epoch = 1000000, lr_method = 'adagrad', x_val = x_val, y_val = y_val)


x_test = utils.load(test_path, mode = 'test', fea_select = fea_select, y_pos = y_pos) 
x_test = utils.scaling(x_test, max, min) 

predicted = lm.predict(x_test, b, w)
print('>>> Predicted Result :\n', predicted)

utils.save_scaler(max, min, scaler_path)
utils.save_model(b, w, model_path)
utils.save_ans(predicted, output_path)

