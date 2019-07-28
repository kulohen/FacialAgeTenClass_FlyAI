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
import sys

from flyai.dataset import Dataset
from flyai.source.base import DATA_PATH
from flyai.source.source import Source
import pandas as pd

# TODO 构建自己的Util



def readCsv_onFlyai(readCsvOnLocal=True):
    try:
        f = open(os.path.join(sys.path[0], 'train.json'))
        line = f.readline().strip()
    except IOError:
        line = ""

    if readCsvOnLocal:
        source_csv = Source().create_instance("flyai.source.csv_source", 'Csv',
                                              {'train_url': os.path.join(DATA_PATH, "dev.csv"),
                                               'test_url': os.path.join(DATA_PATH, "dev.csv")}, line)
    else:
        source_csv = Source().create_instance("flyai.source.csv_source", 'Csv',
                                              {'train_url': os.path.join(DATA_PATH, "train.csv"),
                                               'test_url': os.path.join(DATA_PATH, "test.csv")}, line)
    # 实际上返回的是 flyai.source.csv_source.py的 Csv类, source_csv.data 是train_csv文件,source_csv.val 是test_csv文件
    return source_csv


def readCustomCsv(train_csv_url, test_csv_url):
    try:
        f = open(os.path.join(sys.path[0], 'train.json'))
        line = f.readline().strip()
    except IOError:
        line = ""

    source_csv = Source().create_instance("flyai.source.csv_source", 'Csv',
                                          {'train_url': os.path.join(DATA_PATH, train_csv_url),
                                           'test_url': os.path.join(DATA_PATH, test_csv_url)}, line)

    return source_csv

def ExtendDataFrameToSize(dataframe,size):
    '''

    :param dataframe: 要被扩展的csv
    :param size: 扩展到指定size的大小。
    :return: 比如原csv是12行，size输入25，那么csv至少扩展到25以上，即是12 *2 *2 =48，返回48的大小。
    '''
    if dataframe.shape[0] > size :
        return dataframe
    else:
        dataframe = pd.concat([dataframe,dataframe.copy(deep=False)])
        return ExtendDataFrameToSize(dataframe, size)

def ExtendCsvToSize(source_csv , label='label', size=-1):
    df5 = pd.DataFrame()
    for i in range(10):
        if source_csv[source_csv[label] == i].empty :
            break
        tmp = ExtendDataFrameToSize( source_csv[source_csv[label] == i], size)
        df5 = pd.concat([df5, tmp])
    # print(df5)
    # df5.to_csv(os.path.join(DATA_PATH, 'wangyi-1.csv'), index=False)
    return df5


if __name__=='__main__':
    get_csv = readCsv_onFlyai(True)
    # print(get_csv.data)
    x_train, y_train, x_test, y_test = get_csv.get_all_data()
    print(y_train)

    dataset2 = Dataset(source=readCustomCsv("test_custom.csv", "test_custom.csv"))
    print('dataset2.get_train_length()', dataset2.get_train_length())
    print('dataset2.get_validation_length()', dataset2.get_validation_length())
    xx_train , yy_train= dataset2.next_train_batch()
    # print('yy_train',yy_train)
    # save csv
    df = pd.DataFrame(data=readCustomCsv("test_custom.csv", "test_custom.csv").data)
    # df.to_csv( os.path.join(DATA_PATH, 'wangyi-1.csv'), index=False)
    print(df[df['label']==3])
    # print(df.isin([4, 6]))
    df2=df.copy(deep=False)
    df3 = pd.concat([df,df2])
    print('df3.size',df3.shape)
    print(df3.shape[0])

    df4 =ExtendDataFrameToSize(df,355)
    print('df4.shape',df4.shape)
    # df4.to_csv(os.path.join(DATA_PATH, 'wangyi-1.csv'), index=False)
    print(df[df['label'] == 7])

    df6 =ExtendCsvToSize(df , label='label' ,size=30)
    print('df6.shape',df6.shape)

    df7 = pd.DataFrame(data=readCustomCsv("dev.csv", "dev.csv").data)
    df7 =ExtendCsvToSize(df7 ,size=36)
    print('df7.shape', df7.shape)
    df7.to_csv(os.path.join(DATA_PATH, 'wangyi-2.csv'), index=False)
