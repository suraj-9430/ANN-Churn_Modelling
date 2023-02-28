# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:49:55 2023

@author: rajsu
"""

import numpy as np
import pandas as pd

dataset=pd.read_csv("Churn_Modelling.csv")

x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
x[:,1]=lb.fit_transform(x[:,1])
x[:,2]=lb.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
on=OneHotEncoder(drop="first").fit(x[:,[1]])
xyz=on.transform(x[:,[1]]).toarray()

x=np.append(arr=x,values=xyz.astype(int),axis=1)
x=x[:,[0,2,3,4,5,6,7,8,9,10,11]]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

"""step-2"""
"""initialize the ANN with these libray"""
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
"""first layer=input layer && first hidden layer""" 
classifier.add(Dense(6,activation=("relu"),kernel_initializer="uniform",input_dim=11))

"""second hidden layer && third """
classifier.add(Dense(6,activation=("relu"),kernel_initializer="uniform"))
classifier.add(Dense(6,activation=("relu"),kernel_initializer="uniform"))

"""output layer"""
classifier.add(Dense(1,activation=("sigmoid"),kernel_initializer="uniform"))

classifier.compile(optimizer="adam",loss="BinaryCrossentropy",metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=(12),epochs=200)

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.50)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)