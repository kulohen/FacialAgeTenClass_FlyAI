# -*- coding: utf-8 -*
import argparse
from keras.applications import ResNet50,VGG16,InceptionResNetV2,DenseNet121,DenseNet201
from flyai.dataset import Dataset
from keras.layers import Input,Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
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
from flyai.utils import remote_helper
from time import clock
from processor import img_size

'''
设置项目的超级参数
'''

try:
    weights_path =None
    # weights_path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    # weights_path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.8|densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5")
except OSError:
    weights_path = 'imagenet'



parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()


MODEL_PATH_FILE = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

num_classes = 10
train_epoch = 800
eval_weights_count = 9.8 # 应该是eval_weights的10个求和
history_train = 0
history_test = 0
best_score_by_acc = 0.
best_score_by_loss = 999.
lr_level=0
train_batch_size = {}
val_batch_size = {
    0: 120,
    1: 48,
    2: 58,
    3: 40,
    4: 26,
    5: 38,
    6: 26,
    7: 16,
    8: 12,
    9: 6,
}


# 训练集的每类的batch的量，组成的list
# train_batch_List = [16] * num_classes
train_batch_List = [
    813,
    345,
    427,
    298,
    191,
    264,
    186,
    113,
    108,
    1
]

# wx+b,这是允许分类的loss最低程度，比如class-9 允许loss在1.2
train_allow_loss = [
    -0.3,
    -0.5,
    -0.5,
    -0.5,
    -0.8,
    -0.5,
    -0.8,
    -0.8,
    -0.8,
    -1.2
]

myhistory = wangyi.historyByWangyi()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''

dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# dataset = wangyi.DatasetExtendToSize(False ,train_size=1773,val_size= 572,classify_count=num_classes)
# dataset = wangyi.DatasetExtendToSize(True ,train_size=40,val_size= 40,classify_count=num_classes)
model = Model(dataset)
dataset_wangyi = wangyi.DatasetByWangyi(num_classes)
dataset_wangyi.set_Batch_Size(train_batch_List, val_batch_size)
'''
dataset.get_train_length() : 5866
dataset.get_all_validation_data(): 1956
predict datas :  1956
y_train.sum(): [1773.  729.  891.  618.  399.  568.  394.  241.  204.   49.]
y_val.sum(): [572. 247. 334. 219. 144. 185. 129.  56.  49.  21.]
'''

'''
实现自己的网络机构
'''
time_0 = clock()
# 创建最终模型
Inp = Input((img_size[0], img_size[1], 3))

# base_model = ResNet50(weights=None, input_shape=(224, 224, 3), include_top=False)
base_model = DenseNet201(weights=weights_path, input_tensor=Inp, include_top=False)

# 增加定制层
x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Flatten(name='flatten_1')(x)

# 冻结不打算训练的层。
# print('base_model.layers', len(base_model.layers))
# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)
#
# for layer in base_model.layers[:-33]:
#     layer.trainable = False
    # print(layer)

x = GlobalAveragePooling2D()(x)
# x = Flatten(name='flatten_1')(x)
# x = Dense(2048, activation='relu' )(x)
predictions = Dense(num_classes, activation="softmax")(x)
# 创建最终模型
# model_cnn = keras_model(inputs=base_model.input, outputs=predictions)
model_cnn = keras_model(inputs=Inp, outputs=predictions)

# 输出模型的整体信息
model_cnn.summary()

model_cnn.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
              )

print('keras model,compile, 耗时：%.1f 秒' % (clock() - time_0))

for epoch in range(train_epoch):

    '''
    1/ 获取batch数据
    '''
    x_3, y_3, x_4, y_4, x_5, y_5 = dataset_wangyi.get_Next_Batch()
    if x_3 is None:
        cur_step = str(epoch + 1) + "/" + str(train_epoch)
        print('\n步骤' + cur_step, ': 无batch 跳过此次循环')
        continue
    # 采用数据增强ImageDataGenerator
    datagen = ImageDataGenerator(
        # rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    # datagen.fit(x_train_and_x_val)
    data_iter_train = datagen.flow(x_3, y_3, batch_size=args.BATCH, save_to_dir=None)
    # 打印步骤和训练集/测试集的量
    cur_step = str(epoch + 1) + "/" + str(train_epoch)
    print('\n步骤' + cur_step, ': %d on train, %d on val' % (len(x_3), len(x_4)))
    '''
    2/ 训练train，验证val
    '''
    # history_train = model_cnn.fit(x=x_3, y=y_3, validation_data=(x_4, y_4),
    #                                         batch_size=args.BATCH ,epochs=1,verbose=2
    #                               )
    # print('np.sum(train_batch_List :',np.sum(train_batch_List))
    for_fit_generator_train_steps = int(np.sum(train_batch_List, axis=0) * 2 / args.BATCH) + 1
    print('该epoch的fit_generator steps是 ', for_fit_generator_train_steps)
    history_train = model_cnn.fit_generator(
        generator=data_iter_train,
        steps_per_epoch=for_fit_generator_train_steps,
        validation_data=(x_4, y_4),
        validation_steps=for_fit_generator_train_steps,
        epochs=1,
        verbose=2
    )
    history_train_all = myhistory.SetHistory(history_train)


    sum_loss = 0
    sum_acc = 0
    for iters in range(num_classes):
        if dataset_wangyi.dataset_slice[iters].get_train_length() == 0 or dataset_wangyi.dataset_slice[
            iters].get_validation_length() == 0:
            continue
        history_test = model_cnn.evaluate(
            x=x_5[iters],
            y=y_5[iters],
            batch_size=None,
            verbose=2
        )
        # 不打印了，显示的界面篇幅有限
        print('class-%d __ loss :%.4f , acc :%.4f' % (iters, history_test[0], history_test[1]))
        sum_loss += history_test[0] * val_batch_size[iters]
        sum_acc += history_test[1] * val_batch_size[iters]
        '''
         2.3修改下一个train batch
        '''
        # val-loss 0.7以下不提供batch, 0.7 * 20 =14
        # next_train_batch_size = int(history_test[0] * 20)
        # next_train_batch_size = int(history_test[0] * val_batch_size[iters]+2)
        next_train_batch_size = history_test[0] + train_allow_loss[iters]
        next_train_batch_size = int (next_train_batch_size * val_batch_size[iters])
        if next_train_batch_size > 50:
            train_batch_List[iters] = next_train_batch_size =50
        elif next_train_batch_size < 1:
            train_batch_List[iters] = next_train_batch_size= 1
        else:
            train_batch_List[iters] = next_train_batch_size

    dataset_wangyi.set_Batch_Size(train_batch_List, val_batch_size)
    # sum_loss =sum_loss / np.sum(train_batch_List, axis = 0)
    # sum_acc = sum_acc / np.sum(train_batch_List, axis=0)

    '''
    3/ 保存最佳模型model
    '''
    # save best acc
    if history_train.history['acc'][0] > 0.6 and \
            round(best_score_by_loss, 1) >= round(history_train.history['val_loss'][0], 1):
        model.save_model(model=model_cnn, path=MODEL_PATH, overwrite=True)
        best_score_by_acc = history_train.history['val_acc'][0]
        best_score_by_loss = history_train.history['val_loss'][0]
        print('【保存】了最佳模型by val_loss : %.4f' % best_score_by_loss)
    '''
    4/ 调整学习率和优化模型
    '''
    # 调整学习率，且只执行一次
    if history_train.history['loss'][0] < 0.9 and lr_level == 0:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=RMSprop(lr=0.0005),
                          metrics=['accuracy'])
        print('【学习率】调整为 : 0,0005')
        lr_level = 1
    elif history_train.history['loss'][0] < 0.4 and lr_level == 1:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=1e-4, momentum=0.9, decay=1e-6, nesterov=True),
                          metrics=['accuracy'])
        print('【学习率】调整为 : 1e-4')
        lr_level = 2
    elif history_train.history['loss'][0] < 0.15 and lr_level == 2:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=5e-5, momentum=0.9, nesterov=True),
                          metrics=['accuracy'])
        print('【学习率】调整为 : 5e-5')
        lr_level = 3
    elif history_train.history['loss'][0] < 0.05 and lr_level == 3:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=1e-5, momentum=0.9, nesterov=True),
                          metrics=['accuracy'])
        print('【学习率】调整为 : 1e-5')
        lr_level = 4
    # TODO 新的学习率，还没完成
    # if optimzer_custom.compareHistoryList( history_train_all['loss'] ,pationce= 5 ,min_delta=0.001) :
    #     model_cnn.compile(loss='categorical_crossentropy',
    #                       optimizer=optimzer_custom.get_next() ,
    #                       metrics=['accuracy'])
    #TODO 动态冻结训练层？

print('best_score_by_acc :%.4f' % best_score_by_acc)
print('best_score_by_loss :%.4f' % best_score_by_loss)