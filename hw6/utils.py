import sys
import os
import time
import csv
import pandas as pd
import numpy as np
import pickle as pkl
# import matplotlib.pyplot as plt

def load_test(path):
    raw = csv.reader(open(path))
    data = list(raw)
    data = np.array(data[1:], dtype = np.dtype('int'))
    return data[:, 1:]

def save_ans(data = None, folder_path = './ans', name = 'ans', extension = 'csv', time_mark = True):
    '''
    # 儲存成csv 用
    ## Arguments
    data : 欲儲存的參數
    name : 將檔案儲存在 ans 資料夾底下，名稱為 name 
    extension : 副檔名
    time_mark : 是否要標示時間
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
        save_name = name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    save_path = folder_path + '/' + save_name
    ans = []
    for i in range(len(data)):
        ans.append((i, int(data[i])))

    df_ans = pd.DataFrame(ans, index = None, columns = ['ID', 'Ans'])
    df_ans.to_csv(save_path, index = False)
        
    print('--Save as :', save_path)

def get_time_mark():
    time_mark = str(time.localtime().tm_year)[-2:] + \
    str(time.localtime().tm_mon).zfill(2) + \
    str(time.localtime().tm_mday).zfill(2) + '_' + \
    str(time.localtime().tm_hour).zfill(2) + '-' + \
    str(time.localtime().tm_min).zfill(2) +'-' + \
    str(time.localtime().tm_sec).zfill(2)
    return time_mark

def load_pkl(path):
    file = open(path, 'rb')
    obj = pkl.load(file)
    file.close()
    return obj

def save_pkl(path, obj):
    file = open(path, 'wb')
    pkl.dump(obj, file)
    file.close()

def save_dir(data = None, path = None):
    '''
    # 用路徑儲存
    ## Arguments
    data : 欲儲存的參數
    path : PATH
    '''
    print('\nSaving file...')
    ans = []
    data = data.reshape(-1,)    
    for i in range(len(data)):
        ans.append((i, int(data[i])))

    df_ans = pd.DataFrame(ans, index = None, columns = ['ID', 'Ans'])
    df_ans.to_csv(path, index = False)
        
    print('--Save as :', path)

def load_pkl(path):
    file = open(path, 'rb')
    obj = pkl.load(file)
    file.close()
    return obj

def save_pkl(path, obj):
    file = open(path, 'wb')
    pkl.dump(obj, file)
    file.close()