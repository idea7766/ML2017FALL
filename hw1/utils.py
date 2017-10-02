import numpy as np
import pandas as pd

def rescaling(data):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        max, min = np.amax(data, axis = 0), np.amin(data, axis = 0)
        new_data = (data - min) / (max - min)
    return new_data, max, min

def scaling(data, max, min):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        new_data = (data - min) / (max - min)
    return new_data

def load(path, mode = 'train'): # only for ML2017FALL hw1
    if mode == 'train':
        df_pm25 = pd.read_csv(path, encoding='big5')
        #將 'NR' 改成 0
        arr_pm25_pre = df_pm25.iloc[:,3:].replace('NR', 0).values
        arr_pm25_pre = arr_pm25_pre.astype(float)

        arr_pm25 = np.array([arr_pm25_pre[:18, 0]])
        for i in range(240):
            for j in range(24):
                if i==0 and j == 0:
                    continue
                arr_pm25 = np.append(arr_pm25, [arr_pm25_pre[18*i:18+18*i, j]], axis = 0)
        return arr_pm25

    if mode == 'test':
        df_pm25 = pd.read_csv(path, encoding='big5', header=None)
        #將 'NR' 改成 0
        arr_pm25_pre = df_pm25.iloc[:,2:].replace('NR', 0).values
        arr_pm25_pre = arr_pm25_pre.astype(float)

        # for i in range(240):
        #     for j in range(24):
        #         if i==0 and j == 0:
        #             continue
        #         arr_pm25 = np.append(arr_pm25, [arr_pm25_pre[18*i:18+18*i, j]], axis = 0)
        return arr_pm25_pre

def align(data, n):
    if n <= 0 or None:
        raise('請輸入大於 0 的數值')
    run = data.shape[0] - n + 1 # 毎 n 比當成 1 筆 data
    ls_align_data = []
    for i in range(run):
        ind = 18 * i
        ls_align_data.append(data[ind: ind + 18].flatten())
    new_data = np.asarray(ls_align_data)
    return new_data