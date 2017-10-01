import numpy as np
from numpy.linalg import inv

def LinearRegression(x, y, lr = 0.0001 , epoch = 5, ex_size = 20, lr_method = 'static'):
    '''
    # Linear Regression
    ## Basic Concpet
    model: Linear_function 
    Loss_function: square error
    Opt_algo: SGD
    Learning_Rate: Static
    ## Attirbute
    x: traing data X
    y: traing data Y
    lr: learning rate
    epoch: epoch數
    '''
    if x.ndim != 2:
        raise('= =寫二維陣列啦')
    
    # initialization
    b = 0
    w = np.zeros(x.shape[1])

    if lr_method == 'adagrad':
        # print('目前只有 static learning rate\n')
        print('使用adagrad')
        SGD_dyn_lr(x, y, lr, b, w, epoch)

    elif lr_method == 'static':
        # 數次 epoch 的 SGD, 還沒做 random choice
        for i in range(epoch):
            b, w = SGD(x, y, lr, b, w)
            print('epoch:', i+1)  

    return b, w

def LinearRegression_close(x, y):
    '''
    # 用於驗證
    ## Return
    b: bias
    w: weight array
    '''
    x = np.insert(x, 0, values = 1, axis = 1)
    x_trans = np.transpose(x)
    
    y_mat = np.transpose(np.mat(y))
    # print(y_mat.shape)
    x_mat = np.mat(x)
    # print(x_mat.shape)
    x_trans_mat = np.mat(x_trans)
    # print(x_trans_mat.shape)

    w = inv(x_trans_mat * x_mat)  * x_trans_mat * y_mat
    w = np.array(w)
    w = w.flatten()
    # print(w)

    return w[0], w[1:] #return b, w

def SGD(x, y, lr, b, w):
    '''
    no 'S' here now 
    ## Attribute
    x: example x
    y: example y
    lr: learning rates 
    b: constant bias
    w: weight array

    ## Return 
    b: constant bias
    w: weight array
    '''
    
    w_temp = np.zeros(x.shape[1] + 1)
    w = np.insert(w, 0, b)
    x = np.insert(x, 0, 1, axis = 1)

    num_fea = x.shape[1]
    x_count = x.shape[0]

    gradient_w = gradient(x, y, w)
    
    w_temp = w - (lr * gradient_w * (1 / x_count))

    return w_temp[0], w_temp[1:]

def SGD_dyn_lr(x, y, lr, b, w, epoch):
    '''
    no 'S' here now 
    ## Attribute
    x: example x
    y: example y
    lr: learning rates 
    b: constant bias
    w: weight array

    ## Return 
    b: constant bias
    w: weight array
    '''
    sum_squ_grad = 0
    sta_lr = lr

    # w_temp = np.zeros(x.shape[1] + 1)
    w = np.insert(w, 0, b)
    x = np.insert(x, 0, 1, axis = 1)

    num_fea = x.shape[1]
    x_count = x.shape[0]
    for i in range(epoch):
        gradient_w = gradient(x, y, w)
        ada, sum_squ_grad = adagrad(gradient_w, sum_squ_grad)
        lr = sta_lr / ada
        w = w - lr * gradient_w #* (1 / x_count)

    return w[0], w[1:]

def gradient(x, y, w): 
    '''
    caculus of loss function (square error)
    '''
    num_fea = x.shape[1]

    gradient_weight = np.zeros(num_fea)
    hypothesis = np.dot(x, w)
    loss = hypothesis - y
    print('se: ', sum(loss ** 2))
    # print('hypothesis:', hypothesis)

    x_trans = np.transpose(x)
    gradient_weight = 2 * np.dot(x_trans, loss)
    # print('gradient: ', gradient_weight)

    return gradient_weight

def predcit(x, b, w):
    count = x.shape[0]
    y = np.array([])
    for i in range(count):
        y_const = b + np.dot(w, x[i])
        y = np.append(y, y_const)
    return y

def mse(y, y_pred):
    return sum((y-y_pred) ** 2) *(1 / y.shape[0])

def adagrad(gradient, sum_squ_grad):
    sum_squ_grad += gradient ** 2
    adagrad = sum_squ_grad ** 0.5
    return adagrad, sum_squ_grad 