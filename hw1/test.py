import pandas as pd
import numpy as np
import os
path = os.path.join(os.path.dirname(__file__), "./data/train.csv")

df_pm25 = pd.read_csv(path, encoding='big5')
arr_pm25 = df_pm25.iloc[:,3:].replace('NR', 0).values
arr_pm25 = arr_pm25.astype(float)
arr_pm25 = np.transpose(arr_pm25)
arr_temp = np.array([])
for i in range(arr_pm25.shape[]):
    pass
print(arr_pm25)
print(arr_pm25.shape)
