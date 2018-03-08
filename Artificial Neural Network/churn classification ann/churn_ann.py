# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:01:42 2018

@author: Pool
"""

#引用三个常用的数据处理库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#数据的获取
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#对非数字的分类数据进行处理
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#因为该类的类别有三种，会是0，1，2，需要通过OneHotEncoder来使其变成[0,0,1]
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#对数据进行标准化处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

#来做训练数据和测试数据的分离
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

#建立模型
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#输入输入层和第一隐层
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim= 11))

#输入第二隐层
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))

#输出层
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#编译这个神经网络
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#设置一些假定参数
classifier.fit(X_train, y_train, batch_size= 20, epochs=100)

#预测测试数据值
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)                                  #会产生false和true的bool值数列

#通过 confusion_metix来评估测试值
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)


