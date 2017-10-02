import pandas as pd
import numpy as np
import os

import utils

path = os.path.join(os.path.dirname(__file__), "./data/train.csv")

train_data = utils.load(path)
print(train_data.shape)
train_data = utils.align(train_data, 9)

print(train_data.shape)
