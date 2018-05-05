from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GRU
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py

with h5py.File(''.join(['sp500_close.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value

outputFileName='sp500_close_GRU_tanh_relu_'

step = datas.shape[1]
units= 200
batchSize = 128
features = datas.shape[2]
epochs = 150
outputSize=10

#split training and validation
size = 0.2
trainingSize = int((1-size)* datas.shape[0])
trainingData = datas[:trainingSize,:]
trainingLabels = labels[:trainingSize,:,0]
validationData = datas[trainingSize:,:]
validationLabels = labels[trainingSize:,:,0]

#build model
model = Sequential()
model.add(GRU(units=units, input_shape=(step,features)))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(outputSize))
model.add(Activation('relu'))
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(trainingData, trainingLabels, batchSize=batchSize,validation_data=(validationData,validationLabels), epochs = epochs, callbacks=[CSVLogger(outputFileName+'.csv', append=True),ModelCheckpoint('weights/'+outputFileName+'-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min')])
