import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas as pd
from tabulate import tabulate

seed = 9
np.random.seed(seed)

# Load the Dataset
dataframe = pd.read_csv("ZData.csv", sep=',', skiprows=1, header=None)
dataset = dataframe.values

# Split into input and output
X = dataset[:,0:4]
Y = dataset[:,4]
X_norm = X / X.max(axis=0)
Y_norm = Y / Y.max(axis=0)

[k,l,m,n,o] = [Y.max(axis=0), X[:,0].max(axis=0),X[:,1].max(axis=0),X[:,2].max(axis=0),X[:,3].max(axis=0)]

#Split into testing and training
(X_train, X_test, Y_train, Y_test) = train_test_split(X_norm,Y_norm, test_size = 0.20, random_state=seed)

#Create the model
print('Loading Model.....')
model = Sequential()
model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation = 'linear'))
model.add(Dense(100, kernel_initializer='normal', activation = 'sigmoid'))
model.add(Dense(100, kernel_initializer='normal', activation = 'sigmoid'))
model.add(Dense(1, kernel_initializer='normal'))
print('Model successfully loaded!')

#Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
print('Model Compiled')

#Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=64)

#Predict and Compare the test results
predictions = k*model.predict(X_test)
predictions = np.squeeze(predictions)
col0 = l*np.squeeze(X_test[:,0])
col1 = m*np.squeeze(X_test[:,1])
col2 = n*np.squeeze(X_test[:,2])
col3 = o*np.squeeze(X_test[:,3])
error = abs(np.subtract(predictions,k*Y_test))
results = [col0, col1, col2, col3, predictions, k*Y_test, error]
results = np.transpose(results)
np.savetxt("Predictions_1000.csv", results, delimiter=",",fmt='%.3f')
#To show tabulated data
headers = ["W","H","S","E","Z - Predictions", "Z - Actual Value", "Error"]
table = tabulate(results, headers, tablefmt="fancy_grid")
print(table)
print("Average Error is ", np.mean(error))

#Save the Model
model.save('my_model1000.h5')
print("Saved model to disk")