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
实现自己的网络机构
'''
num_classes = 10
sqeue =ResNet50( weights=None, input_shape=(200, 200, 3), classes= num_classes, include_top=True)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# 输出模型的整体信息
# sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 模型保存的路径
model.check( MODEL_PATH)
MODEL_PATH_FILE = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

# callbacks回调函数的定义
savebestonly = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_loss', mode='auto', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20 ,verbose=1)
xuexilv = ReduceLROnPlateau(monitor='loss',patience=20, verbose=1)


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
data_iter_train = datagen.flow(x_train_and_x_val, y_train_and_y_val, batch_size=args.BATCH , save_to_dir = None, subset='training')
data_iter_validation = datagen.flow(x_train_and_x_val, y_train_and_y_val, batch_size=args.BATCH , save_to_dir = None, subset='validation')
# 验证集可以也写成imagedatagenerator

print('x_train_and_x_val.shape :', x_train_and_x_val.shape)


history = sqeue.fit_generator(
    generator= data_iter_train,
    steps_per_epoch=250,
    validation_data=data_iter_validation,
    validation_steps=1,
    # callbacks = [early_stopping,xuexilv],
    callbacks = [ savebestonly, xuexilv],
    epochs =args.EPOCHS,
    verbose=2,
    workers=24
    # use_multiprocessing=True
)

