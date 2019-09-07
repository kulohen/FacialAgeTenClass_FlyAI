import numpy as np


per = np.random.permutation(train_X.shape[0])		#打乱后的行号
new_train_X = train_X[per, :, :]		#获取打乱后的训练数据
new_train_y = trainy[per]