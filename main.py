# -*- coding: utf-8 -*
import argparse
from keras.applications import ResNet50,VGG16,InceptionResNetV2
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from keras.models import Model as keras_model
from model import Model
from path import MODEL_PATH
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD,Adam
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
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
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

dataset_slice = wangyi.getDatasetListByClassfy(classify_count=num_classes)
x_val_slice,y_val_slice = [],[]
for epoch in range(num_classes):
    x_tmp,y_tmp = dataset_slice[epoch].get_all_validation_data()
    x_val_slice.append(x_tmp)
    y_val_slice.append(y_tmp)

'''
实现自己的网络机构
'''

# sqeue = ResNet50( weights=None, include_top=True, input_shape=(300, 300, 3),classes=num_classes)
sqeue = InceptionResNetV2(weights=None, include_top=True ,classes=num_classes)


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001,epsilon=1e-8)

# 输出模型的整体信息
# sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# 模型保存的路径
# model.check( MODEL_PATH)
MODEL_PATH_FILE = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

# callbacks回调函数的定义
savebestonly = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_loss', mode='auto', save_best_only=True, verbose=0)
save_acc = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_acc', mode='auto', save_best_only=True, verbose=0)

early_stopping = EarlyStopping(monitor='loss', patience=20 ,verbose=1,min_delta=0.001)
xuexilv = ReduceLROnPlateau(monitor='loss',patience=4, verbose=1)


history_train = 0
history_test = 0
best_score_by_acc = 0.
best_score_by_loss = 999.
for epoch in range(args.EPOCHS):
    history_train = sqeue.fit(
        x=x_train,
        y=y_train,
        batch_size=args.BATCH,
        verbose=2,
        callbacks= [ xuexilv,early_stopping],
        # validation_split=0.,
        validation_data=(x_val,y_val),
        shuffle=True
        # class_weight=cw
        # class_weight = 'auto'
    )
    print('learning rate:' ,history_train.history['lr'][0])
    sum_loss = 0.
    sum_acc = 0.
    for iters in range(num_classes):
        history_test = sqeue.evaluate(
            x=x_val_slice[iters],
            y=y_val_slice[iters],
            batch_size=None,
            verbose=2
        )
        print('class-%d __ loss :%.4f , acc :%.4f' %(iters ,history_test[0],history_test[1]))
        sum_loss +=history_test[0]
        sum_acc +=history_test[1]
    # TODO train loss小于 0.7 ，开始保存h5（最佳的val_acc）,同时开始降低学习率
    if history_train.history['loss'][0] >0.7 :
        pass
    else:
        # save best acc
        if best_score_by_acc < sum_acc / num_classes:
            model.save_model(model=sqeue, path=MODEL_PATH, overwrite=True)
            best_score_by_acc = sum_acc / num_classes
            best_score_by_loss = sum_loss / num_classes
            print('保存了最佳模型by val_acc')
    # save best loss
    # if best_score_by_loss > sum_loss/num_classes:
    #     model.save_model(model=sqeue,path=MODEL_PATH,overwrite=True)
    #     best_score_by_loss = sum_loss/num_classes
    #     print('保存了最佳模型by val_loss')


    print('步骤 %d / %d: 自定义10类平均 val_loss is %.4f, val_acc is %.4f\n' %(epoch+1,args.EPOCHS, sum_loss/num_classes , sum_acc/num_classes))

print('best_score_by_acc :%.4f' %best_score_by_acc)
print('best_score_by_loss :%.4f' %best_score_by_loss)