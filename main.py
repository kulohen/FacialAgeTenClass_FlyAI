# -*- coding: utf-8 -*
import argparse
import os
from time import clock

import WangyiUtilOnFlyai as wangyi
import numpy as np
from flyai.dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator
from model import KERAS_MODEL_NAME
from model import Model
from net import Net
from path import MODEL_PATH

'''
设置项目的超级参数
'''

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()


MODEL_PATH_FILE = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

num_classes = 10
train_epoch = 600
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
    5: 39,
    6: 27,
    7: 16,
    8: 13,
    9: 6,
}


# 训练集的每类的batch的量，组成的list
train_batch_List = [50] * num_classes

# wx+b,这是允许分类的loss最低程度，比如class-9 允许loss在1.2
train_allow_loss = [
    -0.0,
    -0.1,
    -0.1,
    -0.1,
    -0.8,
    -0.1,
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
model = Model(dataset)
dataset_wangyi = wangyi.DatasetByWangyi(num_classes)
dataset_wangyi.set_Batch_Size(train_batch_List, val_batch_size)


'''
实现自己的网络机构
'''
time_0 = clock()
# 创建最终模型

model_cnn = Net().get_Model()
# model_cnn = keras_model(inputs=Inp, outputs=predictions)

# 输出模型的整体信息
model_cnn.summary()

model_cnn.compile(loss='categorical_crossentropy',
              optimizer=wangyi.OptimizerByWangyi().get_create_optimizer(name='adam', lr_num=1e-4),
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
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=0.1,
        zoom_range=0.2,
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
        next_train_batch_size = int(history_test[0] * 10)
        # next_train_batch_size = int(history_test[0] * val_batch_size[iters]+2)
        # next_train_batch_size = history_test[0] + train_allow_loss[iters]
        # next_train_batch_size = int (next_train_batch_size * val_batch_size[iters])
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
            round(best_score_by_loss, 2) >= round(history_train.history['val_loss'][0], 2):
    # if history_train.history['acc'][0] > 0.6 and \
    #         round(best_score_by_acc, 2) <= round(history_train.history['val_acc'][0], 2):
        model.save_model(model=model_cnn, path=MODEL_PATH, overwrite=True)
        best_score_by_acc = history_train.history['val_acc'][0]
        best_score_by_loss = history_train.history['val_loss'][0]
        print('【保存】了最佳模型by val_loss : %.4f' % best_score_by_loss)
    '''
    4/ 调整学习率和优化模型
    '''
    tmp_opt = None
    if epoch == 0 or epoch==50 or epoch==100:
        pass
    elif epoch % 50 ==0:
        tmp_opt = wangyi.OptimizerByWangyi().get_random_opt()

    # 调整学习率，且只执行一次
    if history_train.history['loss'][0] < 1.5 and lr_level == 0:

        tmp_opt = wangyi.OptimizerByWangyi().get_create_optimizer(name='adam', lr_num=1e-3)
        lr_level = 1

    elif history_train.history['loss'][0] < 1.0 and lr_level == 1:
        tmp_opt = wangyi.OptimizerByWangyi().get_create_optimizer(name='adam', lr_num=1e-5)
        lr_level = 2

    elif history_train.history['loss'][0] < 0.8 and lr_level == 2:
        tmp_opt = tmp_opt = wangyi.OptimizerByWangyi().get_create_optimizer(name='adagrad', lr_num=1e-4)
        lr_level = 3

    elif history_train.history['loss'][0] < 0.4 and lr_level == 3:
        tmp_opt = tmp_opt = wangyi.OptimizerByWangyi().get_create_optimizer(name='adagrad', lr_num=1e-5)
        lr_level = 4

    # 应用新的学习率
    if tmp_opt is not None:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=tmp_opt,
                          metrics=['accuracy'])

    # TODO 新的学习率，还没完成
    # if optimzer_custom.compareHistoryList( history_train_all['loss'] ,pationce= 5 ,min_delta=0.001) :
    #     model_cnn.compile(loss='categorical_crossentropy',
    #                       optimizer=optimzer_custom.get_next() ,
    #                       metrics=['accuracy'])
    #TODO 动态冻结训练层？

    '''
    5/ 冻结训练层
    '''

print('best_score_by_acc :%.4f' % best_score_by_acc)
print('best_score_by_loss :%.4f' % best_score_by_loss)