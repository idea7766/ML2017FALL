import os
import random 
import time
import numpy as np
import pandas as pd

base_dir = os.path.dirname(__file__)

def load_data(path):
    data = pd.read_csv(path)
    data = data.values
    return data

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

def load_scaler(path):
    scaler = np.load(path)
    return scaler[0], scaler[1]

def validation(feats, lables, ratio = 0.1):
    '''
    # Arguments
    ratio : 多少比例的資料要被當作 validation set 
    # Return
    X_train: , Y_train: , X_val: , Y_val:
    '''
    nb_cut = int(feats.shape[0] *ratio)

    x_train = feats[:-nb_cut]
    y_train = lables[:-nb_cut]

    x_val = feats[-nb_cut:]
    y_val = lables[-nb_cut:]

    return x_train, y_train, x_val, y_val

def shuffle(x, y):
    '''
    # shuffle x 和 y
    '''
    fea_num = x.shape[1]
    data = np.insert(x, fea_num, y, axis =1)
    np.random.shuffle(data)
    x = data[:, :fea_num]
    y = data[:, fea_num:]
    return x, y.flatten()

def save_model(b, w, path, time_mark = True):
    if w.ndim != 1:
        raise('不支援 ndim != 1 的arr')
    if time_mark:
        path = path + '_' + \
        str(time.localtime().tm_year)[-2:] + \
        str(time.localtime().tm_mon).zfill(2) + \
        str(time.localtime().tm_mday).zfill(2) + '_' + \
        str(time.localtime().tm_hour).zfill(2) + '-' + \
        str(time.localtime().tm_min).zfill(2) +'-' + \
        str(time.localtime().tm_sec).zfill(2)
    b_w = np.insert(w, 0, b) # bias 放第 0 項
    np.save(path, b_w)
    print(">>> model 儲存存成功")

def load_model(path):
    w = np.load(path)
    return w[0], w[1:]

def save_ans(data = None, name = 'ans', extension = 'csv', time_mark = True):
    '''
    # 儲存成csv 用
    ## Arguments
    data : 欲儲存的參數
    name : 將檔案儲存在 ans 資料夾底下，名稱為 name 
    extension : 副檔名
    time_mark
    '''
    print('\nSaving file...')
    if time_mark:
        save_name = name + '_' + \
        str(time.localtime().tm_year)[-2:] + \
        str(time.localtime().tm_mon).zfill(2) + \
        str(time.localtime().tm_mday).zfill(2) + '_' + \
        str(time.localtime().tm_hour).zfill(2) + '-' + \
        str(time.localtime().tm_min).zfill(2) +'-' + \
        str(time.localtime().tm_sec).zfill(2) + '.' +\
        extension

    else:
        save_name = name + extension
    
    save_path = base_dir + '/ans/' + save_name
    ans = []
    for i in range(len(data)):
        ans.append((i+1, int(data[i])))

    df_ans = pd.DataFrame(ans, index = None, columns = ['id', 'label'])
    df_ans.to_csv(save_path, index = False)
        
    print('--Save as :', save_path)

def save_ans_dir(data , path):
    save_path = path
    ans = []
    for i in range(len(data)):
        ans.append((i+1, int(data[i])))

    df_ans = pd.DataFrame(ans, index = None, columns = ['id', 'label'])
    df_ans.to_csv(save_path, index = False)
        
    print('--Save as :', save_path)

def save_scaler(max, min, path):
    scaler = np.append([max], [min], axis = 0)
    np.save(path, scaler)
    print('>>> scaler 儲存成功')