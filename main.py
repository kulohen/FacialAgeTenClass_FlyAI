# -*- coding: utf-8 -*
import argparse
from keras.applications import ResNet50,VGG16,InceptionResNetV2
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization,LeakyReLU
from keras.models import Model as keras_model
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
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper

try:
    weights_path = remote_helper.get_remote_date(
        "https://www.flyai.com/m/v0.7|inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5")
except OSError:
    weights_path = 'imagenet'
'''
2019-07-26
获取数据值，是否train set有问题？？读取label
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
num_classes = 10
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# dataset = wangyi.DatasetExtendToSize(False ,train_size=1773,val_size= 572,classify_count=num_classes)
# dataset = wangyi.DatasetExtendToSize(True ,train_size=40,val_size= 40,classify_count=num_classes)
model = Model(dataset)
'''
dataset.get_train_length() : 5866
dataset.get_all_validation_data(): 1956

'''
print('dataset.get_train_length()',dataset.get_train_length())
print('dataset.get_validation_length()',dataset.get_validation_length())
x_train, y_train , x_val, y_val =dataset.get_all_processor_data()

'''
实现自己的网络机构
'''

# sqeue = ResNet50( weights=None, include_top=True, input_shape=(300, 300, 3),classes=num_classes)
sqeue = InceptionResNetV2(weights=None, include_top=True, input_shape=(299, 299, 3),classes=num_classes)

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
savebestonly = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_loss', mode='auto', save_best_only=True, verbose=0)
save_acc = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_acc', mode='auto', save_best_only=True, verbose=0)

early_stopping = EarlyStopping(monitor='loss', patience=20 ,verbose=1,min_delta=0.003)
xuexilv = ReduceLROnPlateau(monitor='loss',patience=4, verbose=1)




history = sqeue.fit(
    x=x_train,
    y=y_train,
    batch_size=args.BATCH,
    epochs=args.EPOCHS,
    verbose=2,
    callbacks= [ savebestonly, save_acc, xuexilv,early_stopping],
    # validation_split=0.,
    validation_data=(x_val,y_val),
    shuffle=True
    # class_weight=cw
    # class_weight = 'auto'
)


# print(history.history)

