# -*- coding: utf-8 -*
import argparse
from keras.applications import ResNet50,VGG16
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization
from keras.models import Sequential
from model import Model
from path import MODEL_PATH
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD,adam
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import sys
import os
from model import KERAS_MODEL_NAME
import WangyiUtilOnFlyai as wangyi
'''
2019-07-26
获取数据值，是否train set有问题？？读取label
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset2 = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
dataset = wangyi.DatasetExtendToSize(False ,train_size=1773,val_size= 572,classify_count=10)
# dataset = wangyi.DatasetExtendToSize(True ,train_size=40,val_size= 30,classify_count=10)
model = Model(dataset)
'''
dataset.get_train_length() : 5866
dataset.get_all_validation_data(): 1956

'''
x_train, y_train , x_val, y_val =dataset.get_all_processor_data()
print('dataset.get_train_length() :',dataset.get_train_length())
print('dataset.get_all_validation_data():',dataset.get_validation_length())
# 针对one-hot实现的分类统计
print('y_train.sum():',y_train.sum(axis=0))
print('y_val.sum():',y_val.sum(axis=0))
'''
实现自己的网络机构
'''
num_classes = 10
sqeue =ResNet50( weights=None, input_shape=(200, 200, 3), classes= num_classes, include_top=True)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# 输出模型的整体信息
# sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 模型保存的路径
model.check( MODEL_PATH)
MODEL_PATH_FILE = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

# callbacks回调函数的定义
savebestonly = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_loss', mode='auto', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=20 ,verbose=1,min_delta=0.03)
xuexilv = ReduceLROnPlateau(monitor='loss',patience=5, verbose=1)


x_train_and_x_val = np.concatenate((x_train, x_val),axis=0)
y_train_and_y_val= np.concatenate((y_train , y_val),axis=0)

# 采用数据增强ImageDataGenerator
datagen= ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.02,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.25
)
# datagen.fit(x_train_and_x_val)
data_iter_train = datagen.flow(x_train_and_x_val, y_train_and_y_val, batch_size=args.BATCH , save_to_dir = None)
data_iter_validation = datagen.flow(x_train_and_x_val, y_train_and_y_val, batch_size=args.BATCH , save_to_dir = None, subset='validation')
# 验证集可以也写成imagedatagenerator

print('x_train_and_x_val.shape :', x_train_and_x_val.shape)
print('y_train_and_y_val.sum():',y_train_and_y_val.sum(axis=0))
'''
history = sqeue.fit_generator(
    generator= data_iter_train,
    steps_per_epoch=4,
    validation_data=(x_val,y_val),
    validation_steps=1,
    # callbacks = [early_stopping,xuexilv],
    class_weight= 'auto',
    callbacks = [ savebestonly, xuexilv,early_stopping],
    epochs =args.EPOCHS,
    verbose=2,
    workers=6
    # use_multiprocessing=True
)
'''
# 设置不同比例权重
cw = {
    0:33,
    1:80,
    2:64,
    3:93,
    4:144,
    5:104,
    6:150,
    7:263,
    8:309,
    9:1117
}


history = sqeue.fit(
    x=x_train,
    y=y_train,
    batch_size=args.BATCH,
    epochs=args.EPOCHS,
    verbose=2,
    callbacks= [ savebestonly, xuexilv,early_stopping],
    # validation_split=0.,
    validation_data=(x_val,y_val),
    shuffle=True,
    # class_weight=cw
    class_weight = 'auto'
)


# print(history.history)

