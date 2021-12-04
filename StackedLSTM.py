# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:37:35 2021

@author: User
"""



import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

df=pd.read_csv('AuctionFinalData.csv')
df.shape
df.head(5)
print(df)
#plt.plot(df)
plt.figure(figsize = (40,15))
plt.plot(df['Date'],(df['UnitPrice']))
plt.xticks(rotation='vertical')
plt.xlabel('Date',fontsize=8)
plt.ylabel('Coconut Price',fontsize=18)
plt.show()

training_set = df.iloc[:450, 1:2].values
test_set = df.iloc[450:, 1:2].values


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 80 time-steps and 1 output
X_train = []
y_train = []
for i in range(20, len(training_set)):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# #Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
#Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mse',metrics=['accuracy'] )
# Fitting the RNN to the Training set
history=model.fit(X_train, y_train, epochs = 150, batch_size = 32)

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
#plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['X_train', 'y_train'], loc='upper left')
plt.show()


score=model.evaluate(X_train, y_train, batch_size = 32,verbose=1)
#rms = sqrt(score[1])
print(' mean_squared_error:', score)
#print(' root_mean_squared_error:', rms)

# Getting the predicted 
dataset_train = df.iloc[:450, 1:2]
dataset_test = df.iloc[430:, 1:2]
#print(dataset_train)
#print(dataset_test.shape )


dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 20:].values
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(20, len(dataset_test)):
    X_test.append(inputs[i-20:i, 0])
X_test = np.array(X_test)
#print(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#print(X_test.shape)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print(predicted_stock_price)
predicted_stock_price.shape

# Visualising the results
plt.figure(figsize = (30,15))
#plt.plot(df.loc[:450, 'Date'],(df.loc[:450,'UnitPrice']),color = 'black', label = 'Real Coconut Price')
plt.plot(df.loc[430:, 'Date'],dataset_test.values, color = 'red', label = 'Real Coconut Price')
#plt.plot(df.loc[370:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted Coconut Price')
#plt.xticks(np.arange(0,90,100))
#plt.plot(dataset_test.values, color = 'red', label = 'Real Coconut Price')
plt.plot(df.loc[450:, 'Date'],predicted_stock_price, color = 'blue', label = 'Stacked LSTM Predicted Coconut Price')
plt.plot(df.loc[450:, 'Date'],vpredicted_stock_price, color = 'Green', label = 'Vanilla LSTM Predicted Coconut Price')
plt.plot(df.loc[450:, 'Date'],bipredicted_stock_price, color = 'Orange', label = 'Bi Directional LSTM Predicted Coconut Price')
plt.xticks(rotation='vertical')
plt.title('Coconut Price Prediction')
plt.xlabel('Time')
plt.ylabel('Coconut Price')
plt.legend()
plt.show()