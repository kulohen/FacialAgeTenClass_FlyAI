import numpy as np
import WangyiUtilOnFlyai as wangyi
import pandas as pd

# from keras.applications import ResNet50,VGG16,InceptionResNetV2,DenseNet121,DenseNet201
# base_model = DenseNet121( weights=None,input_shape=(224,224, 3), include_top=False)
# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)


x =[
    [0,1,2,3,4,5,6,7,8,9],
    [00,11,22,33,44,55,66,77,88,99],
    [000,111,222,333,444,555,666,777,888,999],
    [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
    ]

np.random.shuffle(x)
print(x)
# np.random.choice(x,2,replace=False)
print(x[:2])

