# -*- coding: utf-8 -*
import argparse
from flyai.dataset import Dataset
from model import Model
from flyai.utils.yaml_helper import Yaml
import os
from path import MODEL_PATH

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
x_train, y_train , x_val, y_val =dataset.get_all_processor_data()
print('dataset.get_train_length() :',dataset.get_train_length())
print('dataset.get_all_validation_data():',dataset.get_validation_length())
'''
这里的y_train已经是one-hot形式，没法统计了
或者针对one-hot来统计
'''
print('y_train.shape:',y_train.shape)
print('y_train.argmax(1):', y_train[5].argmax())

# 针对one-hot实现的分类统计
print('y_train.sum():',y_train.sum(axis=0))
print('y_val.sum():',y_val.sum(axis=0))
# print('y_train.count(1):',y_train[0:100][0].count(1))

# yaml = Yaml(path=os.path.join(MODEL_PATH, "app.yaml")).processor()
model.predict_all(dataset.evaluate_data_no_processor())