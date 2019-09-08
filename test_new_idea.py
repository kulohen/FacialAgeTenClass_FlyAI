import numpy as np
import WangyiUtilOnFlyai as wangyi
import pandas as pd

try:
    source_csv = wangyi.readCustomCsv_V3('train.csv', 'test.csv')
    print('train.csv , test.csv 读取成功')
except:
    print('train.csv , test.csv 读取失败')
    source_csv = None

if source_csv is None:
    try:
        source_csv = wangyi.readCustomCsv_V3('dev.csv', 'dev.csv')
        print('dev.csv 读取成功')
    except:
        print('train.csv,test.csv,dev.csv 都读取失败')

# step 1 : csv to dataframe
dataframe_train = pd.DataFrame(data=source_csv.c.data)
dataframe_test = pd.DataFrame(data=source_csv.c.val)

# TODO train and test merge one, and split by myself
newone = pd.concat([dataframe_train, dataframe_test], axis=0)
newtwo = newone.sample(frac = 1)
print(len(newtwo))
cut_length = int(len(newtwo)*0.8)
print(len(newtwo[:cut_length]))
print(len(newtwo[cut_length:]))

