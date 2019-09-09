import numpy as np
import WangyiUtilOnFlyai as wangyi
import pandas as pd
from keras.applications import ResNet50,VGG16,InceptionResNetV2,DenseNet121,DenseNet201
base_model = DenseNet121( weights=None,input_shape=(224,224, 3), include_top=False)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
