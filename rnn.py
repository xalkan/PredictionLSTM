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



# Part 2 - Building the RNN

# Part 3 - Making the predictions and visualising the results
