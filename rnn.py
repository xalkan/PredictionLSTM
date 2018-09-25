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

# Part 3 - Making the predictions and visualising the results
