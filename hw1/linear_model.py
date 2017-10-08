import numpy as np
from numpy.linalg import inv

def LinearRegression(x, y, lr = 0.0001 , epoch = 10000, lr_method = 'static', x_val = None,  y_val = None):
    '''
    # Linear Regression
    ## Basic Concpet
    model: Linear_function 
    Loss_function: Square error
    Opt_algo: Gradient descnt
    Learning_Rate: Static

    ## Attirbute
    x: traing data X
    y: traing data Y
    lr: learning rate
    epoch: i.e. iteration 的回數
    lr_method: 調整 learning rate 的方法，有 'static' 及 'adagrad'
    x_val: 若有 validation set，則會作 early stopping
    y_val: 若有 validation set，則會作 early stopping 

    ## Return
    b: constant bias
    w: weight array
    '''
    if x.ndim != 2:
        raise('= =寫二維陣列啦')
    
    # initialization
    b = 0
    w = np.zeros(x.shape[1])

    if lr_method == 'adagrad':
        # print('目前只有 static learning rate\n')
        print('使用adagrad')
        b, w  = SGD_dyn_lr(x, y, lr, b, w, epoch, x_val = x_val, y_val = y_val)

    elif lr_method == 'static' and x_val != None and y_val !=None:
        # 數次 epoch 的 SGD, 還沒做 random choice
        loss = None
        for i in range(epoch):
            b, w = SGD(x, y, lr, b, w)
            print('epoch:', i+1)
            if i % 1000 ==0:
                stop, loss = early_stopping(x_val, y_val, b, w, loss)
                if  stop == True:
                    print('>>>break at epoch :', i)
                    break
    else:
        for i in range(epoch):
            print('epoch:', i+1)
            b, w = SGD(x, y, lr, b, w)
            
    return b, w

def LinearRegression_close(x, y):
    '''
    # 用於驗證的 close form
    ## Return
    b: bias
    w: weight array
    '''
    x = np.insert(x, 0, values = 1, axis = 1)
    x_trans = np.transpose(x)
    
    y_mat = np.transpose(np.mat(y))
    # y_mat = np.mat(y)  
    # print('y_mat.shape', y_mat.shape)
    x_mat = np.mat(x)
    # print('x_mat.shape', x_mat.shape)
    x_trans_mat = np.mat(x_trans)
    # print('x_trans_mat.shape', x_trans_mat.shape)

    w = inv(x_trans_mat * x_mat)  * x_trans_mat * y_mat
    w = np.array(w)
    w = w.flatten()
    print(w)

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

def SGD_dyn_lr(x, y, lr, b, w, epoch, x_val = None, y_val = None):
    '''
    no 'S' here now 
    ## Attribute
    x: example x
    y: example y
    lr: learning rates 
    b: constant bias
    w: weight array
    x_val: 若有 validation set，則會作 early stopping
    y_val: 若有 validation set，則會作 early stopping    

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
    if x_val != None and y_val != None:
        loss = None
        for i in range(epoch):            
            gradient_w = gradient(x, y, w, i=i)
            ada, sum_squ_grad = adagrad(gradient_w, sum_squ_grad)
            lr = sta_lr / ada
            w = w - lr * gradient_w
            if i % 1000 ==0:
                stop, loss = early_stopping(x_val, y_val, w[0], w[1:], loss)
                if  stop == True:
                    print('>>>break at epoch :', i)
                    break    
    else:
        for i in range(epoch):
            gradient_w = gradient(x, y, w)
            ada, sum_squ_grad = adagrad(gradient_w, sum_squ_grad)
            lr = sta_lr / ada
            w = w - lr * gradient_w

    return w[0], w[1:]

def gradient(x, y, w, i): 
    '''
    caculus of loss function (square error)
    '''
    num_fea = x.shape[1]

    gradient_weight = np.zeros(num_fea)
    hypothesis = np.dot(x, w)
    # print(hypothesis.shape)
    loss = hypothesis - y    
    print('>>> epoch : %d \t  | square error: %f' %(i, sum(loss ** 2)))
    # print('hypothesis:', hypothesis)

    x_trans = np.transpose(x)
    gradient_weight = 2 * np.dot(x_trans, loss)
    # print('gradient: ', gradient_weight)

    return gradient_weight

def predict(x, b, w):
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

def early_stopping(x, y, b, w, pre_loss):
    if pre_loss == None:
        return False, mse(y, predict(x, b, w))
    stop = False
    loss = mse(y, predict(x, b, w))
    if loss >= pre_loss:
        stop = True
    return stop, loss