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



if __name__=='__main__':
    get_csv = readCsv_onFlyai(True)
    # print(get_csv.data)
    x_train, y_train, x_test, y_test = get_csv.get_all_data()
    print(y_train)

    dataset2 = Dataset(source=readCustomCsv("test_custom.csv", "test_custom.csv"))
    print('dataset2.get_train_length()', dataset2.get_train_length())
    print('dataset2.get_validation_length()', dataset2.get_validation_length())
    xx_train , yy_train= dataset2.next_train_batch()
    print('yy_train',yy_train)