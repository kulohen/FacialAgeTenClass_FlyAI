# -*- coding: utf-8 -*
import argparse
from keras.applications import ResNet50,VGG16,InceptionResNetV2
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from keras.models import Model as keras_model
from model import Model
from path import MODEL_PATH
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD,Adam,RMSprop
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
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
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
predict datas :  1956
y_train.sum(): [1773.  729.  891.  618.  399.  568.  394.  241.  204.   49.]
y_val.sum(): [572. 247. 334. 219. 144. 185. 129.  56.  49.  21.]
'''
x_train, y_train , x_val, y_val =dataset.get_all_processor_data()
x_train_and_x_val = x_train
y_train_and_y_val= y_train

dataset_slice = wangyi.getDatasetListByClassfy(classify_count=num_classes)
x_val_slice,y_val_slice = [],[]
for epoch in range(num_classes):
    x_tmp,y_tmp = dataset_slice[epoch].get_all_validation_data()
    x_val_slice.append(x_tmp)
    y_val_slice.append(y_tmp)
    if epoch==4 or epoch==6 or epoch==7 or epoch==8 or epoch==9 :
        x_train_and_x_val = np.concatenate((x_train_and_x_val, x_tmp),axis=0)
        y_train_and_y_val= np.concatenate((y_train_and_y_val , y_tmp),axis=0)
print('x_train_and_x_val.shape ',x_train_and_x_val.shape)
print('y_train_and_y_val.shape ',y_train_and_y_val.shape)
'''
实现自己的网络机构
'''

# sqeue = ResNet50( weights=None, include_top=True, input_shape=(300, 300, 3),classes=num_classes)
sqeue = InceptionResNetV2(weights=None, include_top=True ,classes=num_classes)


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.003,epsilon=1e-8)

# 输出模型的整体信息
# sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])

# 模型保存的路径
# model.check( MODEL_PATH)
MODEL_PATH_FILE = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

# callbacks回调函数的定义
savebestonly = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_loss', mode='auto', save_best_only=True, verbose=0)
save_acc = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_acc', mode='auto', save_best_only=True, verbose=0)

early_stopping = EarlyStopping(monitor='loss', patience=20 ,verbose=1,min_delta=0.001)
xuexilv = ReduceLROnPlateau(monitor='loss',patience=4, verbose=1)

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

cw_train = {
    0:1,
    1:2.5,
    2:2,
    3:3,
    4:0.3,
    5:3.3,
    6:0.3,
    7:0.3,
    8:0.3,
    9:0.1
}
eval_weights = {
    0:3,
    1:1.2,
    2:1.5,
    3:1,
    4:0.7,
    5:0.9,
    6:0.7,
    7:0.4,
    8:0.3,
    9:0.1
}
eval_weights_count = 9.8 # 应该是eval_weights的10个求和
history_train = 0
history_test = 0
best_score_by_acc = 0.
best_score_by_loss = 999.
lr_level=0



for epoch in range(args.EPOCHS):
    # history_train = sqeue.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=args.BATCH,
    #     verbose=2,
    #     callbacks= [ xuexilv,early_stopping],
    #     # validation_split=0.,
    #     validation_data=(x_val,y_val),
    #     shuffle=True,
    #     class_weight=cw_train
    #     # class_weight = 'auto'
    # )
    history_train = sqeue.fit_generator(
        generator= data_iter_train,
        steps_per_epoch=200,
        validation_data=(x_val,y_val),
        validation_steps=1,
        class_weight= cw_train,
        epochs =1,
        verbose=2,
        workers=6
)
    # print('learning rate:' ,history_train.history)
    # 没有叠加history.查看history的shape，history是叠加的？还是单独1条。用以决定fit()中initial_epoch 是否启用？
    # print('history_train.history len : ', history_train.history['lr'])
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
        # if iters ==0 or iters==1 or iters==2 or iters==3 or iters==5 :
        #     sum_loss +=history_test[0]
        #     sum_acc +=history_test[1]
        sum_loss += history_test[0] * eval_weights[iters]
        sum_acc += history_test[1] * eval_weights[iters]
    #  train loss小于 0.7 (ln0.5)，开始保存h5（最佳的val_acc）,同时开始降低学习率
    if history_train.history['loss'][0] >0.7 :
        pass
    else:
        # save best acc
        if best_score_by_loss > sum_loss/eval_weights_count :
            model.save_model(model=sqeue, path=MODEL_PATH, overwrite=True)
            best_score_by_acc = sum_acc / eval_weights_count
            best_score_by_loss = sum_loss / eval_weights_count
            print('【保存】了最佳模型by val_loss : %.4f' %best_score_by_loss)
    # save best loss
    # if best_score_by_loss > sum_loss/num_classes: best_score_by_acc <  sum_acc / eval_weights_count and
    #     model.save_model(model=sqeue,path=MODEL_PATH,overwrite=True)
    #     best_score_by_loss = sum_loss/num_classes
    #     print('保存了最佳模型by val_loss')

    # 调整学习率，且只执行一次
    if history_train.history['loss'][0] <0.7 and lr_level==0:
        sqeue.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.001),
                      metrics=['accuracy'])
        print('【学习率】调整为 : 0,001')
        lr_level = 1
    elif history_train.history['loss'][0] <0.3 and lr_level==1:
        sqeue.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.00033),
                      metrics=['accuracy'])
        print('【学习率】调整为 : 0,00033')
        lr_level = 2
    elif history_train.history['loss'][0] <0.1 and lr_level==2:
        sqeue.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.0001),
                      metrics=['accuracy'])
        print('【学习率】调整为 : 0,0001')
        lr_level = 3
    elif history_train.history['loss'][0] < 0.05 and lr_level==3:
        sqeue.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-5),
                      metrics=['accuracy'])
        print('【学习率】调整为 : 1e-5')
        lr_level = 4

    print('步骤 %d / %d: 自定义 val_loss is %.4f, val_acc is %.4f\n' %(epoch+1,args.EPOCHS, sum_loss/eval_weights_count , sum_acc/eval_weights_count))

print('best_score_by_acc :%.4f' %best_score_by_acc)
print('best_score_by_loss :%.4f' %best_score_by_loss)