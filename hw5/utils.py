import sys
import time
import csv
import pandas as pd
import numpy as np
import pickle as pkl

###################################
# -----------NEW UTILS----------- #
###################################

def load_data(path, file_type = 'train'):
    if file_type == 'train':
        raw = csv.reader(open(path))
        data = list(raw)
        data = np.array(data[1:], dtype = np.dtype('int'))
        x = data[:, 1:3]
        y = data[:, 3]
        return x, y
    elif file_type == 'test':
        raw = csv.reader(open(path))
        data = list(raw)
        data = np.array(data[1:], dtype = np.dtype('int'))
        x = data[:, 1:3]
        return x

def post_process(y, round = True, bias = 0):
    y += bias
    if round:
        y = np.around(y)
    y[y < 1] = 1
    y[y > 5] = 5
    return y

def data_split(x, y, grab_per_usr = 1):
    if grab_per_usr < 1:
        grab_per_usr = 1
    train = np.insert(x, 0, y, axis = 1)
    max_usr_id = int(np.amax(x[:,0]))
    min_usr_id = int(np.amin(x[:,0]))
    train_ls = np.array([[0]*3])
    val_ls = np.array([[0]*3])
    for i in range(min_usr_id, max_usr_id+1):   
        usr_i = train[train[:,1] == i]
        len_usr = len(usr_i)
        if len_usr >= grab_per_usr +1:
            ran_ind = np.arange(len_usr)
            np.random.shuffle(ran_ind)
            usr_i = usr_i[ran_ind]
            train_ls = np.concatenate((train_ls, usr_i[grab_per_usr:]))
            val_ls = np.concatenate((val_ls, usr_i[:grab_per_usr]))
        elif len_usr < grab_per_usr and len_usr >= 1:
            ran_ind = np.arange(len_usr)
            np.random.shuffle(ran_ind)
            usr_i = usr_i[ran_ind]
            grab = len_usr - 1
            train_ls = np.concatenate((train_ls, usr_i[grab:]))
            val_ls = np.concatenate((val_ls, usr_i[:grab]))
        elif len_usr >= 0:
            train_ls = np.concatenate((train_ls, usr_i))            
        else:
            continue
    print('length of original training data\t', len(train))
    print('length of new training data\t', len(train_ls) -1)
    print('length of new validation data\t', len(val_ls) -1)
    x_train_ls = train_ls[1:, 1:]
    y_train_ls = train_ls[1:, 0]
    x_val_ls = val_ls[1:, 1:]
    y_val_ls = val_ls[1:, 0]

    return x_train_ls, y_train_ls, x_val_ls, y_val_ls

def find_type_movie(path):
    data = load_with_split(path, '::')
    # num_mov = len(data[1:])
    type_of_mov = []

    for mov in data[1:]:
        type_ls = mov[2].split('|')
        for type_i in type_ls:
            if type_i not in type_of_mov:
                type_of_mov.append(type_i)
    save_pkl('mov.pkl', type_of_mov)
    print(type_of_mov)

def type_movie(path, num_mov = 3953):
    data = load_with_split(path, '::')
    # num_mov = len(data[1:])+1
    type_of_mov = []
    mov_ind = np.zeros((num_mov,))

    for mov in data[1:]:
        mov_id = int(mov[0])
        type_ls = mov[2].split('|')
        for type_i in type_ls:
            if type_i not in type_of_mov:
                type_of_mov.append(type_i)
        mov_ind[mov_id] = type_of_mov.index(type_ls[0])
    # save_pkl('mov.pkl', type_of_mov)
    print(type_of_mov)    
    print(mov_ind)

def load_with_split(path, char = '::'):
    f = open(path, encoding = 'latin-1')
    f = f.readlines()
    # print(f)
    ls = [line.strip().split(char) for line in f]
    return ls
###################################
# -----------UNI UTILS----------- #
###################################
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
    
    save_path = folder_path + '/' + save_name
    ans = []
    data = data.reshape(-1,)
    for i in range(len(data)):
        ans.append((i+1, data[i]))

    df_ans = pd.DataFrame(ans, index = None, columns = ['TestDataID', 'Rating'])
    df_ans.to_csv(save_path, index = False)
        
    print('--Save as :', save_path)

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
        ans.append((i+1, data[i]))

    df_ans = pd.DataFrame(ans, index = None, columns = ['TestDataID', 'Rating'])
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

def shuffle(x, y):
    ran_ind = np.arange(len(x))
    np.random.shuffle(ran_ind)
    x = x[ran_ind]
    y = y[ran_ind]
    return x, y

def rescaling(data):
    '''
    Arrange all feature btw [-1, 1]
    '''
    max, min, mean = np.amax(data, axis = 0), np.amin(data, axis = 0), np.mean(data, axis = 0)
    new_data = (data - mean) / (max - min)
    print(mean)
    return new_data, max, min, mean

def scaling(data, max, min, mean):
    '''
    Arrange all feature btw [-1, 1]
    '''
    new_data = (data - mean) / (max - min)
    print(mean)
    return new_data

