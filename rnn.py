# Recurrent Neural Network - LSTM



# Part 1 - Data Preprocessing

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the training set
# training and testing on completely separate unknown data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# converting the dataframe into numpy array of one column
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling (normalization, not standarizaion, because sigmoid in output layer)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
# apply scaler
training_set_scaled = scaler.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output, 60 previous data rows
# past 60 timesteps, t - 60, just right to avoid under or overfitting
# X_train is past 60 rows, y_train is t + 1 prediction
X_train = []
y_train = []

for i in range(60, len(training_set_scaled + 1)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# convert X_train and y_train from lists to np arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping, adding another dimension
# adding another indicator
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] , 1))


# Part 2 - Building the RNN

# import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialize rnn as sequence of layers (not a computational graph)
regressor = Sequential()

# adding LSTM layers
# units - number of LSTM cells or memory units, lower number of units cant capture more info
# return_sequences - set to true because we're building a stacked LSTM with several
# layers, when finished adding layers, don't include this because its default value
# is false
# input_shape - shape of X_train training set (observation, timesteps, indicators)
# only need to add timesteps and indicators because observatios will automatically
# be taken into account, no need to be explicit
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
# adding dropout regularization to avoid overfitting - classical rate is 0.2 = 20%
regressor.add(Dropout(rate = 0.2))      

# adding more layers, no need to specify input_shape now
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(rate = 0.2))      

regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(rate = 0.2))      

regressor.add(LSTM(units = 50 ))
regressor.add(Dropout(rate = 0.2))      

# adding output layer
regressor.add(Dense(units = 1))

# compiling the rnn, using adam instead of rmsprop because better results in this case
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the rnn to the training data
regressor.fit(x = X_train, y = y_train, batch_size = 32, epochs = 1)

















# Part 3 - Making the predictions and visualising the results
