# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset                                                              
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data                                                  #用imputer来处理数据中空缺的data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


'''
	one-hot:
	注意： one-hot化是每个feature的取值个数作为one-hot后的长度，
	对应的位置0/1的值是原数据在此位置取值的有无.不是取值的max的长度编号:
	如这个最大为6,按照one-hot的定义是testdata[[‘age’]]=[4]–>one-hot后:[0,0,0,1,0,0]
	但定义直观理解是这样的,但在工程实现这样没必要,只针对有值得取值编码就
	可以,节约矩阵one-hot后的规模而且效果一样;
'''


'''
#对于非数值的项，直接用label-encoder 就可以了
'''


'''
其实如果我们跳出 scikit-learn， 在 pandas 中可以很好
地解决这个问题，用 pandas 自带的get_dummies函数即可
pd.get_dummies(testdata,columns=testdata.columns)
'''
