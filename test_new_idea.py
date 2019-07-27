# -*- coding: utf-8 -*
import argparse
from flyai.dataset import Dataset
from model import Model
import numpy as np
from flyai.utils.yaml_helper import Yaml
import os
from path import MODEL_PATH
from flyai.utils import read_data
from flyai.source.source import Source
from flyai.source.base import Base, DATA_PATH
import sys
from flyai.source.csv_source import Csv

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
'''
dataset.get_train_length() : 5866
dataset.get_all_validation_data(): 1956

'''
x_train, y_train, x_val, y_val = dataset.get_all_processor_data()
print('dataset.get_train_length() :', dataset.get_train_length())
print('dataset.get_all_validation_data():', dataset.get_validation_length())
'''
这里的y_train已经是one-hot形式，没法统计了
或者针对one-hot来统计
'''
print('x_train:', x_train.shape)
print('y_train.shape:', y_train.shape)
print('y_train.argmax(1):', y_train[5].argmax())

# 针对one-hot实现的分类统计
print('y_train.sum():', y_train.sum(axis=0))
print('y_val.sum():', y_val.sum(axis=0))
# print('y_train.count(1):',y_train[0:100][0].count(1))

# yaml = Yaml(path=os.path.join(MODEL_PATH, "app.yaml")).processor()
# model.predict_all(dataset.evaluate_data_no_processor())
'''
这里测试随机取样
'''
array1 = np.array([11, 22, 33, 44, 55, 66, 77, 88, 99])
array2 = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999])
print('array1.shape[0] :', array1.shape)
batch_size = 5
slice = np.random.choice(a=array1.shape[0], size=batch_size, replace=False)
print(array1[slice])
print(array2[slice])

random_slice = np.random.choice(a=y_train.shape[0], size=2, replace=False)
print('y_train.shape[0]', y_train.shape[0])
print('random_slice', random_slice)
# print(x_train[random_slice])
# print(y_train[random_slice])


# TODO 构建自己的验证集
'''
撰写 ：  提取y_val 的每类10个，组成100个array的y_val_normalize
用csv创建验证集




for data_per in range(y_val.shape[1]):
    if np.argmax(y_val[data_per])
    print(data_per)
'''


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


get_csv = readCsv_onFlyai(True)
print(get_csv.data)
x_train, y_train, x_test, y_test = get_csv.get_all_data()
print(y_train)
