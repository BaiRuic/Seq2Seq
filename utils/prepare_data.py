import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
import random
from sklearn import preprocessing
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))) # 是当前目录
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) # 是上级目录
import utils.my_logging as my_logging


path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))    # 是上一级目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.DoubleTensor')

# 定义日志实例
mylog = my_logging.My_Logging()

# 将序列数据转化为监督模型可训练的数据
def data_to_supervised(data, time_steps, horizion, features = 2):
    column_data = [] # 存放数据
    column_name = [] # 存放column名字
    features = data.shape[1] # 特征个数
    for i in range(time_steps, 0, -1):
        column_data.append(data.shift(periods=i, axis=0))
        if i !=1:
            column_name += ['load(t-'+str(i-1) +')', 'temp(t-' + str(i-1) + ')']
        else:
            column_name += ['load(t)', 'temp(t)']
    for i in range(0,horizion):
        column_data.append(data['load'].shift(periods=-i, axis=0))
        column_name += ['load(t+' + str(i+1) + ')']
    reframed_data = pd.concat(column_data, axis=1)
    reframed_data.columns = column_name
    reframed_data.dropna(how='any', axis=0,inplace=True)
    return reframed_data

# 数据标准化
def standardizeData(X, SS=None, train = False):
    '''
    Given a list of input features, standizes them to bring them onto a homsgenous scale
    
    Args:
        X([Dataframe]): [A dataframe of all the input values]
        SS([object],optional): [A standardScaler object that hold mean and std of a standardized dataset] Default is None
        train([bool],optional):[if False, mean validation set to be loaded and SS need to be passed to scale it] Default is False
    '''
    if train:
        SS = StandardScaler()
        new_X = SS.fit_transform(X)
        return (new_X, SS)
    else:
        new_X = SS.fit_transform(X)
        return (new_X, None)

# 加载数据及数据预处理
def prepare_data(time_steps=7, horizion=3, features=2):
    mylog.info_logger('prepare data')
    data = pd.read_csv(path + '\\data\\energy.csv',header=0,index_col=0)
    reframed_data = data_to_supervised(data=data, time_steps=time_steps, horizion=horizion, features = 2)
    
    mylog.info_logger("standardize data")
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    stdData, _ = standardizeData(X=reframed_data, SS=scaler, train = False)

    mylog.info_logger('split data into input and output')
    inputs = stdData[:,:-horizion]
    outputs = stdData[:,-horizion:]
    mylog.info_logger(f'inputs.shape:{inputs.shape}')
    mylog.info_logger(f'outputs.shape:{outputs.shape}')

    # 将训练集测试集验证集合分开
    num_examples = inputs.shape[0]
    # 设置各数据集大小
    train_size = int( num_examples * 0.8 )
    valid_size = int( num_examples * 0.1 )

    # 将训练集输入[sample_num, time_steps*features] reshape 为 [sample_num, time_steps, features]
    train_ip, train_op = inputs[:train_size, :], outputs[:train_size]
    train_ip = train_ip.reshape(train_ip.shape[0], time_steps, features)

    valid_ip, valid_op = inputs[train_size:train_size+valid_size, :],outputs[train_size:train_size+valid_size]
    valid_ip = valid_ip.reshape(valid_ip.shape[0], time_steps, features)

    test_ip,test_op = inputs[train_size+valid_size:, :],outputs[train_size+valid_size:]
    test_ip = test_ip.reshape(test_ip.shape[0], time_steps, features)

    mylog.info_logger('train:')
    mylog.info_logger(f'train_ip.shape:{train_ip.shape}')
    mylog.info_logger(f'train_op.shape:{train_op.shape}')
    mylog.info_logger('valid:')
    mylog.info_logger(f'valid_ip.shape:{valid_ip.shape}')
    mylog.info_logger(f'valid_op.shape:{valid_op.shape}')
    mylog.info_logger('test:')
    mylog.info_logger(f'test_ip.shape:{test_ip.shape}')
    mylog.info_logger(f'test_op.shape:{test_op.shape}')

    return train_ip, train_op, test_ip, test_op
    
# 训练集 Dataset
class My_Train_Datasets(torch.utils.data.Dataset):
    def __init__(self,train_ip,train_op):
        super(My_Train_Datasets,self).__init__()
        self.train_input = train_ip
        self.train_output = train_op
        self.len = train_ip.shape[0]

    def __getitem__(self,idx):
        return torch.tensor(self.train_input[idx]), torch.tensor(self.train_output[idx])

    def __len__(self):
        return self.len

# 测试集 Dataset
class My_Test_Dataset(torch.utils.data.Dataset):
    def __init__(self,test_ip, test_op):
        super(My_Test_Dataset, self).__init__()

        self.test_ip = test_ip
        self.test_op = test_op
        self.len = test_ip.shape[0]

    def __getitem__(self, idx):

        return torch.tensor(self.test_ip[idx]), torch.tensor(self.test_op[idx])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_ip, train_op, test_ip, test_op = prepare_data(time_steps=7, horizion=3, features=2)
    train_dataset = My_Train_Datasets(train_ip, train_op)
    test_dataset = My_Test_Dataset(test_ip, test_op)

    # 提取 负载，剔除温度数据
    input_main = train_ip[:,:,0]
    mylog.info_logger(f'train_ip 前两个样本{train_ip[0:3]}')
    mylog.info_logger(f'input_main:{input_main.shape}')
    mylog.info_logger(f'input_main:{input_main.reshape(input_main.shape[0], input_main.shape[1],1)[0:3]}')
    

    