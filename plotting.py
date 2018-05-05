from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, Activation, Reshape, GRU
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from  sklearn.metrics import mean_squared_error
from cycler import cycler
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import h5py

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

with h5py.File(''.join(['sp500_close_test_data_200.h5']), 'r') as hf:
    data = hf['inputs'].value
    labels = hf['outputs'].value
    inputTimes = hf['inputTimes'].value
    outputTimes = hf['outputTimes'].value
    originalInputs = hf['originalInputs'].value
    originalOutputs = hf['originalOutputs'].value
    originalData = hf['originalData'].value

#test data
size = 0.38
testData = data[int((1-size)* data.shape[0]):,:,:]
testOutputTimes = outputTimes[int((1-size)* data.shape[0]):,:,:]

df = pd.read_csv('data/sp500rohkemTest.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

step = data.shape[1]
features = data.shape[2]
#print(step, features)
outputSize=10
units= 200

model = Sequential()
model.add(GRU(units=units, input_shape=(step,features)))
model.add(Activation('tanh'))
model.add(Dense(outputSize))
model.add(Activation('relu'))
model.load_weights('weights/sp500_close_GRU_tanh_relu_-121-0.00032.hdf5')
model.compile(loss='mse', optimizer='adam')
model.summary()
originalData.shape

predicted = model.predict(testData)
predictedInverted = []

scaler=MinMaxScaler()
scaler.fit(originalData[:,0].reshape(-1,1))
predictedInverted.append(scaler.inverse_transform(predicted))
#print (np.array(predictedInverted).shape)

predictedInverted = np.array(predictedInverted)[0,:,:].reshape(-1)
#print (np.array(predictedInverted).shape)
testOutputTimes = pd.to_datetime(testOutputTimes.reshape(-1), unit='s')

predictionDataFrame = pd.DataFrame()
predictionDataFrame['Times'] = testOutputTimes
predictionDataFrame['Value'] = predictedInverted
predictionDataFrame.shape

df = df.tail(int(len(predictionDataFrame.Times)/outputSize+outputSize))

tradeDatesDataFrame = pd.read_csv('data/pmi_dates.csv')
tradeDatesDataFrame['Date'] = pd.to_datetime(tradeDatesDataFrame['Date'], unit='s')
tradeDatesList = tradeDatesDataFrame['Date'].values.tolist()

predictedValues = []
lastPredictedValues = []

plt.plot(df['Timestamp'], df['Close'], label = 'Actual', linewidth = 3.0, color = 'black')
plt.legend(loc='upper left')
for i in range(int(len(predictionDataFrame.Times)/outputSize)):
    plt.plot(predictionDataFrame.ix[i*outputSize:i*outputSize+outputSize-1].Times,predictionDataFrame.ix[i*outputSize:i*outputSize+outputSize-1].Value)
    lastPredictedValues.append(predictionDataFrame.ix[i*outputSize+outputSize-1].Value)
    if predictionDataFrame.ix[i*outputSize].Times.value in tradeDatesList:
        predictedValues.append(predictionDataFrame.ix[i*outputSize:i*outputSize+outputSize-1].Value)
plt.show()

print('MSE : ' , mean_squared_error(df.tail(int(len(predictionDataFrame.Times)/outputSize))['Close'],lastPredictedValues))
print('MAPE : ' , mean_absolute_percentage_error(df.tail(int(len(predictionDataFrame.Times)/outputSize))['Close'],lastPredictedValues))

colormap = plt.cm.bwr
normalize = mlt.colors.Normalize(vmin=-0, vmax=1)
plt.scatter(df.tail(int(len(predictionDataFrame.Times)/outputSize))['Close'],lastPredictedValues, cmap=colormap, norm=normalize, marker='*')
plt.scatter(df.tail(int(len(predictionDataFrame.Times)/outputSize))['Close'],df.tail(int(len(predictionDataFrame.Times)/outputSize))['Close'], color = 'black', s=0.6)
plt.show()


sp500Data = 'data/sp500_data.xlsx'
sp500 = pd.read_excel(sp500Data, delimiter=';', index_col='Date', header=0)

portfolio = pd.DataFrame(index=sp500.index)
portfolio['Open'] = sp500['Open']
portfolio['Close'] = sp500['Close']
portfolio['Signal'] = 0.0
averagePositionOpenDays = 0
initialCapital = 10000

for countingIndex in range(len(tradeDatesList)):
    biggestDiffList = []
    for i in predictedValues[countingIndex]:
        biggestDiffList.append(float(i - sp500.loc[sp500.index == tradeDatesDataFrame['Date'].iloc[countingIndex]].Open))
    if max(biggestDiffList,key=abs) > 0:
        portfolio.loc[portfolio.index == tradeDatesDataFrame['Date'].iloc[countingIndex], 'Signal'] = 1.0
        portfolio.iloc[portfolio.index.get_loc(tradeDatesDataFrame['Date'].iloc[countingIndex])+1+biggestDiffList.index(max(biggestDiffList,key=abs)), 2] = -1.0
        averagePositionOpenDays += biggestDiffList.index(max(biggestDiffList,key=abs))
    else:
        portfolio.loc[portfolio.index == tradeDatesDataFrame['Date'].iloc[countingIndex], 'Signal'] = -1.0
        portfolio.iloc[portfolio.index.get_loc(tradeDatesDataFrame['Date'].iloc[countingIndex])+1+biggestDiffList.index(max(biggestDiffList,key=abs)), 2] = 1.0
        averagePositionOpenDays += biggestDiffList.index(max(biggestDiffList,key=abs))
    
numberOfTrades = len(portfolio.loc[portfolio['Signal'] != 0.0])/2
print('Tehinguid tehti : ', numberOfTrades)

portfolio['Original_Signal'] = portfolio['Signal']

openPositions = 0
shortPositions = 0
averageReturn = 0
for date in portfolio.index:
    if portfolio.at[date, 'Signal'] != 0.0:
        openPositions += 1
        if openPositions % 2 == 1:
            openPosition = portfolio.at[date, 'Open']
            positionDirection = portfolio.at[date, 'Signal']
            if portfolio.at[date, 'Signal'] == -1:  
                shortPositions += 1
        if openPositions % 2 == 0:
            closePosition = portfolio.at[date, 'Close']
            if positionDirection == -1:
                averageReturn += 100 * (openPosition - closePosition)/ openPosition
            else: averageReturn += 100 * (closePosition - openPosition)/ openPosition
    if portfolio.at[date, 'Signal'] == 0.0 and openPositions % 2 == 1:
        portfolio.at[date, 'Signal'] = portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal']

print('Short positions : ', shortPositions)
print('Keskmine tehingu tootlus : ', (averageReturn / numberOfTrades))
print('Keskmine tehingu pikkus : ', (averagePositionOpenDays / numberOfTrades))
    
portfolio['Holdings'] = portfolio['Signal']*portfolio['Open']
portfolio['Cash'] = initialCapital - portfolio['Holdings']

for date in portfolio.index:
    if portfolio.at[date, 'Signal'] == portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal'] or portfolio.at[date, 'Signal'] == -portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal']:
        portfolio.at[date, 'Holdings'] = portfolio.at[date, 'Signal'] * portfolio.at[date, 'Close']
    if portfolio.at[date, 'Signal'] == portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal'] and portfolio.at[date, 'Signal'] != 0.0:
        portfolio.at[date, 'Cash'] = portfolio.iloc[portfolio.index.get_loc(date)-1]['Cash']
    if portfolio.at[date, 'Signal'] == -portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal'] and portfolio.at[date, 'Signal'] != 0.0:
        portfolio.at[date, 'Cash'] = portfolio.iloc[portfolio.index.get_loc(date)-1]['Cash'] - portfolio.at[date, 'Holdings']
        portfolio.at[date, 'Holdings'] = 0.0
    if portfolio.iloc[portfolio.index.get_loc(date)-2]['Signal'] == -portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal'] and portfolio.at[date, 'Signal'] == 0.0:
        portfolio.at[date, 'Cash'] = portfolio.iloc[portfolio.index.get_loc(date)-1]['Cash']
    if portfolio.at[date, 'Signal'] == portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal']:
        portfolio.at[date, 'Cash'] = portfolio.iloc[portfolio.index.get_loc(date)-1]['Cash']
    if portfolio.at[date, 'Signal'] != 0.0 and portfolio.iloc[portfolio.index.get_loc(date)-1]['Signal'] == 0.0:
        portfolio.at[date, 'Cash'] = portfolio.iloc[portfolio.index.get_loc(date)-1]['Cash'] - portfolio.at[date, 'Holdings']


portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']
        

fig = plt.figure()
ax1 = fig.add_subplot(111,  ylabel='Price in $')
sp500['Close'].plot(ax=ax1, color='r', lw=2.)
ax1.plot(portfolio.Close[portfolio.Original_Signal >= 1.0],
        '^', markersize=10, color='g')
ax1.plot(portfolio.Close[portfolio.Original_Signal <= -1.0],
        'v', markersize=10, color='k')


fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
portfolio['Total'].plot(ax=ax1, lw=2.)
ax1.plot(portfolio.Total[portfolio.Original_Signal == 1.0],
         '^', markersize=10, color='g')
ax1.plot(portfolio.Total[portfolio.Original_Signal == -1.0],
         'v', markersize=10, color='k')

plt.show()
