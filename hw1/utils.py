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

def load(path, mode = 'train', fea_select = None): # only for ML2017FALL hw1
    if mode == 'train':
        df_pm25 = pd.read_csv(path, encoding='big5')
        #將 'NR' 改成 0
        arr_pm25_pre = df_pm25.iloc[:,3:].replace('NR', 0).values
        arr_pm25_pre = arr_pm25_pre.astype(float)

        arr_pm25 = np.array([arr_pm25_pre[:18, 0]])

        day_num = int(arr_pm25[0] / 18)
        hour_of_day = 24
        for i in range(day_num):
            for j in range(hour_of_day):
                if i==0 and j == 0:
                    continue
                data_of_hour = arr_pm25_pre[18*i:18+18*i, j]
                arr_pm25 = np.append(arr_pm25, [data_of_hour], axis = 0)
        
        if fea_select != None:
            arr_pm25 = arr[:, fea_select]
        arr_pm25 = align(arr_pm25, 9)
        return arr_pm25

    if mode == 'test':
        df_pm25 = pd.read_csv(path, encoding='big5', header=None)
        #將 'NR' 改成 0
        arr_pm25_pre = df_pm25.iloc[:,2:].replace('NR', 0).values
        arr_pm25_pre = arr_pm25_pre.astype(float)
        arr_pm25 = np.array([np.transpose(arr_pm25_pre[:18]).flatten()])

        data_num = int(arr_pm25_pre.shape[0]/18)
        for i in range(run):
            if i == 0:
                continue
            data_of_9hr = arr_pm25, [np.transpose(arr_pm25_pre[18*i:18*i+18]
            arr_pm25 = np.append(data_of_9hr).flatten()], axis=0)

        return arr_pm25

def align(data, num):
    if num <= 0 or None:
        raise('請輸入大於 0 的數值')

    run = data.shape[0] - num + 1 # 毎 n 比當成 1 筆 data
    ls_align_data = np.array([data[0:num].flatten()])
    for ind in range(run):
        if ind == 0:
            continue
        ls_align_data = np.append(ls_align_data, [data[ind:ind+num].flatten()], axis = 0)
    new_data = np.asarray(ls_align_data)
    return new_data

def write_out_ans(data, path):
    col = ['id', 'value']
    ans_sheet = []
    for i in range(data.shape[0]):
        ans_sheet.append(('id_' + str(i), int(data[i])))
    df_ans = pd.DataFrame(ans_sheet, index = None, columns = col)
    df_ans.to_csv(path, index=False)
    print("應該存成功了...")