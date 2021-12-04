# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:59:07 2021

@author: User SAMEERA 18880242
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# Importing Coconut data from CSV

df=pd.read_csv('AuctionFinalData.csv')


training_set = df.iloc[:450, 1:2].values
test_set = df.iloc[450:, 1:2].values


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 20 time-steps and 1 output
X_train = []
y_train = []
for i in range(20, len(training_set)):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = Sequential()
model.add(Bidirectional(LSTM(50, input_shape=(X_train.shape[1], 1))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history=model.fit(X_train, y_train, epochs = 150, batch_size = 32)

score=model.evaluate(X_train, y_train, batch_size = 32,verbose=1)
print(' mean_squared_error:', score)


model.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['X_train', 'y_train'], loc='upper left')
plt.show()


dataset_train = df.iloc[:450, 1:2]
dataset_test = df.iloc[430:, 1:2]


dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(20, len(dataset_test)):
    X_test.append(inputs[i-20:i, 0])
X_test = np.array(X_test)
#print(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


bipredicted_stock_price = model.predict(X_test)
bipredicted_stock_price = sc.inverse_transform(bipredicted_stock_price)


bipredicted_stock_price.shape

# Visualising the results
plt.figure(figsize = (50,15))

plt.plot(df.loc[430:, 'Date'],dataset_test.values, color = 'red', label = 'Real Coconut Price')
plt.plot(df.loc[450:, 'Date'],bipredicted_stock_price, color = 'blue', label = 'Predicted Coconut Price')
plt.xticks(rotation='vertical')
plt.title('Coconut Price Prediction')
plt.xlabel('Time')
plt.ylabel('Coconut Price')
plt.legend()
plt.show()

# Print Data
print((df.loc[450:, 'Date']
       ,bipredicted_stock_price))