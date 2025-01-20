import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

tf.random.set_seed(7)

# Load the data
filename = '../airline-passengers.csv'
df = pd.read_csv('../airline-passengers.csv', usecols=[1], engine='python')
dataset = df.values.astype('float32')
print(df.head())

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# Train and test split sets
train_size = int(len(dataset_scaled) * 0.67)
test_size = len(dataset_scaled) - train_size
train = dataset_scaled[0:train_size,:]
test = dataset_scaled[train_size:len(dataset_scaled),:]
print(f'Train: {len(train)}, Test: {len(test)}')

def create_lag(data, lag_value=1):
	dataX = np.array(data)
	shiftY = pd.DataFrame(data).shift(-lag_value).values
	dataY = np.reshape(shiftY, len(dataX))
	# Remove NaN values
	dataX[np.isnan(dataX)] = 0
	dataY[np.isnan(dataY)] = 0
	return dataX, dataY

# Reshape with lag so X=t and Y=t+1
lag_value = 1
trainX, trainY = create_lag(train, lag_value)
testX, testY = create_lag(test, lag_value)
print(f'Train: {len(testX)}, Train Y: {len(testY)}')

# Reshape input as [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM model network
model = Sequential()
model.add(LSTM(4, input_shape=(1, lag_value)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print("Prediction: ", testPredict[0])

# Re-scale predictions by inverting transformation
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Metric calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Plot train and test predictions
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lag_value:len(trainPredict)+lag_value, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict):len(dataset), :] = testPredict

# Plot predictions against dataset
plt.plot(scaler.inverse_transform(dataset_scaled), color='black')
plt.plot(trainPredictPlot, color='green')
plt.plot(testPredictPlot, color='red')
plt.title("Predictions and Actual Data")
plt.show()