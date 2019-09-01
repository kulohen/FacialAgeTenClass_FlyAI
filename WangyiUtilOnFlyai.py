#!/usr/bin/env python
# coding:utf-8
"""
Name : WangyiUtilOnFlyai.py
Author  : 莫须有的嚣张
Contect : 291255700
Time    : 2019/7/28 上午9:42
Desc:
"""

import os
from time import clock

import keras.optimizers as optmzs
import numpy as np

import pandas as pd
from flyai.core import Lib
from flyai.dataset import Dataset

from flyai.source.base import DATA_PATH
from flyai.source.csv_source import Csv


lr_level = {
            0:0.001,
            1:0.0003,
            2:0.0001,
            3:3e-5,
            4:1e-5
        }
optimizer_level = {
    0: 'sgd',
    1: 'rmsprop',
    2: 'adagrad',
    3: 'adadelta',
    4: 'adam',
    5: 'adamax',
    6: 'nadam'
}
optimizer_name = {
            'sgd' : optmzs.SGD,
            'rmsprop': optmzs.RMSprop,
            'adagrad' : optmzs.Adagrad,
            'adadelta' : optmzs.Adadelta,
            'adam' : optmzs.Adam,
            'adamax' : optmzs.Adamax,
            'nadam' : optmzs.Nadam
        }
class OptimizerByWangyi():
    def __init__(self, pationce=5 , min_delta =0.003):
        self.optimizer_iterator = 0
        self.lr_iterator = 0
        self.pationce_count = 0

    def get_create_optimizer(self,name=None,lr_num=0):
        if name==None or lr_num==0:
            raise ValueError('请指定正确的优化器/学习率')

        x = optimizer_name[name](lr=lr_num)
        print('采用了优化器： ',name , ' --学习率: ', lr_num)
        return x

    def get_next(self, optimzer = None, lr = None):

        if optimzer is not None:
            name_1 = optimzer
        else:
            name_1 = optimizer_level[self.optimizer_iterator]

        if lr is not None:
            lr_1 = lr
        else:
            lr_1 = lr_level[self.lr_iterator]

        x = self.get_create_optimizer(name_1 , lr_1 )

        self.lr_iterator = (self.lr_iterator + 1) % len(lr_level)
        if self.lr_iterator == 0 :
            self.optimizer_iterator = (self.optimizer_iterator +1 ) % len(optimizer_level)
        return x
    #TODO 写降低学习率的回调功能？
    def compareHistoryList(self, loss_list =None , pationce = 10, min_delta =0.001):
        '''
        判断这个自定义的callback，达到earlystopping , reduceLearnRate 的条件 返回True
        :param loss_list:
        :param pationce:
        :param min_delta:
        :return:
        '''
        self.pationce_count += 1
        if loss_list is None :
            ValueError('第一个参数不能为空')
        elif len(loss_list) < pationce or self.pationce_count <pationce:
            return  False
        #TODO 写上判断的逻辑，或者叫做callback

        # elif  (loss_list[-pationce]*pationce -sum(loss_list[-pationce: ]) ) <\
        elif (sum(loss_list[-pationce:-pationce/2]) - sum(loss_list[-pationce/2:])) < \
                  (min_delta * pationce) :
            print('um(loss_list[-pationce:-pationce/2]) - sum(loss_list[-pationce/2:])) <(min_delta * pationce ：',
                  sum(loss_list[-pationce:-pationce/2]) ,' - ',sum(loss_list[-pationce/2:]),' < ' , min_delta * pationce)
            self.pationce_count = 0
            return True
        else:
            return False


def readCustomCsv_V3(train_csv_url, test_csv_url):
    # 2019-08-29 flyai改版本了，这是为了适应


    source_csv = Csv({'train_url': os.path.join(DATA_PATH, train_csv_url),
                                           'test_url': os.path.join(DATA_PATH, test_csv_url)})
    return source_csv


def get_sliceCSVbyClassify_V2(label='label',classify_count=3):
    # 2019-08-29 flyai改版本了，这是为了适应
    try:
        source_csv=readCustomCsv_V3('train.csv', 'test.csv')
        print('train.csv , test.csv 读取成功')
    except:
        print('train.csv , test.csv 读取失败')
        source_csv = None

    if source_csv is None:
        try:
            source_csv = readCustomCsv_V3('dev.csv', 'dev.csv')
        except:
            print('train.csv,test.csv,dev.csv 都读取失败')


    # step 1 : csv to dataframe
    dataframe_train = pd.DataFrame(data=source_csv.c.data)
    dataframe_test = pd.DataFrame(data=source_csv.c.val)

    # step 2 : 筛选 csv


    list_path_train,list_path_test = [],[]
    for epoch in range(classify_count):
        path_train = os.path.join(DATA_PATH, 'wangyi-train-classfy-' + str(epoch) + '.csv')
        dataframe_train[dataframe_train[label] == epoch].to_csv(path_train,index=False)
        list_path_train.append(path_train)

        path_test = os.path.join(DATA_PATH, 'wangyi-test-classfy-' + str(epoch) + '.csv')
        dataframe_test[dataframe_test[label] == epoch].to_csv(path_test,index=False)
        list_path_test.append(path_test)
        print('classfy-',epoch,' : train and test.csv save OK!')

    return list_path_train,list_path_test


def getDatasetListByClassfy_V4(classify_count=3):
    # 2019-08-29 flyai改版本了，这是为了适应

    xx, yy = get_sliceCSVbyClassify_V2(classify_count=classify_count)
    list_tmp=[]
    for epoch in range(classify_count):
        time_0 = clock()
        dataset = Lib(source=readCustomCsv_V3(xx[epoch], yy[epoch]), epochs=1)
        list_tmp.append(dataset)
        # print('class-',epoch,' 的flyai dataset 建立成功')
        print('class-', epoch, ' 的flyai dataset 建立成功, 耗时：%.1f 秒' % (clock() - time_0), '; train_length:',
              dataset.get_train_length(), '; val_length:', dataset.get_validation_length())

    return list_tmp

class historyByWangyi():
    '''
    总结main.py中使用的代码，规整成1个类，方便调用
    '''
    def __init__(self):
        # 自定义history
        self.history_train_all = {}
        self.history_train_loss = []
        self.history_train_acc = []
        self.history_train_val_loss = []
        self.history_train_val_acc = []

    def SetHistory(self,history_train):

        self.history_train_loss.append(history_train.history['loss'][0])
        self.history_train_acc.append(history_train.history['acc'][0])
        self.history_train_val_loss.append(history_train.history['val_loss'][0])
        self.history_train_val_acc.append(history_train.history['val_acc'][0])
        self.history_train_all['loss'] = self.history_train_loss
        self.history_train_all['acc'] = self.history_train_acc
        self.history_train_all['val_loss'] = self.history_train_val_loss
        self.history_train_all['val_acc'] = self.history_train_val_acc

        return self.history_train_all

class DatasetByWangyi():
    def __init__(self, n):
        self.num_classes= n

        time_0 = clock()
        self.dataset_slice = getDatasetListByClassfy_V4(classify_count=n)
        self.optimzer_custom = OptimizerByWangyi()
        print('全部分类的flyai dataset 建立成功, 耗时：%.1f 秒' % (clock() - time_0))

        # 平衡输出45类数据
        self.x_3, self.y_3, self.x_4, self.y_4 = [], [], [], []
        self.x_5, self.y_5 ,self.x_6,self.y_6= {}, {} , {}, {}
        self.train_batch_List = []
        self.val_batch_size = {}

    def set_Batch_Size(self,train_size,val_size):
        self.train_batch_List = train_size
        self.val_batch_size = val_size

    def get_Next_Batch(self):
        # 平衡输出45类数据
        x_3, y_3, x_4, y_4 = [], [], [], []
        x_5, y_5 = {}, {}

        for iters in range(self.num_classes):
            if self.dataset_slice[iters].get_train_length() == 0 or self.dataset_slice[
                iters].get_validation_length() == 0 or self.train_batch_List[iters] == 0:
                continue
            xx_tmp_train, yy_tmp_train, xx_tmp_val, yy_tmp_val = self.dataset_slice[iters].next_batch(
                size=self.train_batch_List[iters], test_size=self.val_batch_size[iters])
            # 合并3类train
            x_3.append(xx_tmp_train)
            y_3.append(yy_tmp_train)
            x_4.append(xx_tmp_val)
            y_4.append(yy_tmp_val)
            x_5[iters] = xx_tmp_val
            y_5[iters] = yy_tmp_val

        # 跳出当前epoch，貌似不需要这个if
        if len(x_3) == 0 or len(y_3) == 0 or len(x_4) == 0 or len(y_4) == 0:
            return None
        x_3 = np.concatenate(x_3, axis=0)
        y_3 = np.concatenate(y_3, axis=0)
        x_4 = np.concatenate(x_4, axis=0)
        y_4 = np.concatenate(y_4, axis=0)
        return x_3, y_3, x_4, y_4 ,x_5, y_5





if __name__=='__main__':
    time_0 = clock()
    dataset = Dataset()
    print('常规的flyai dataset 建立成功, 耗时：%.1f 秒' % (clock() - time_0))

    num_classes = 45
    start_lr = 0.001

    print('dataset.get_train_length()', dataset.get_train_length())
    print('dataset.get_validation_length()', dataset.get_validation_length())

    print(dataset.lib)

    '''
    读取csv
    '''
    train_csv_url = test_csv_url = 'dev.csv'

    source_csv = Csv({'train_url': os.path.join(DATA_PATH, train_csv_url),
                      'test_url': os.path.join(DATA_PATH, test_csv_url)})

    '''
    调用csv存储
    '''


    print(source_csv)
    dataset_slice = getDatasetListByClassfy_V4(45)