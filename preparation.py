import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py
 
class PastSampler:
    'Forms training samples for predicting future values from past value'
     
    def __init__(self, N, K, slidingWindow = True):
        'Predict K future sample using N previous samples'
        self.K = K
        self.N = N
        self.slidingWindow = slidingWindow
 
    def transform(self, A):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #indexes
        if self.slidingWindow:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M)+np.arange(0,A.shape[0],M).reshape(-1,1)
                
            else:
                I = np.arange(M)+np.arange(0,A.shape[0] -M,M).reshape(-1,1)
            
        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]    #Number of features per sample
        return B[:, :ci], B[:, ci:] #Sample matrix, Target matrix

#data file path
dfp = 'data/sp500rohkemTest.csv'

#Columns of price data to use
columns = ['Close']
df = pd.read_csv(dfp)
timeStamps = df['Timestamp']
df = df.loc[:,columns]
originalDataFrame = pd.read_csv(dfp).loc[:,columns]


fileName='sp500_close.h5'

scaler = MinMaxScaler()
#normalization
for c in columns:
    df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))
    
#Features are input sample dimensions(channels)
A = np.array(df)[:,None,:]
original_A = np.array(originalDataFrame)[:,None,:]
timeStamps = np.array(timeStamps)[:,None,None]

#Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 200, 10         #Number of past and future samples
ps = PastSampler(NPS, NFS, slidingWindow=True)
B, Y = ps.transform(A)
inputTimes, outputTimes = ps.transform(timeStamps)
original_B, original_Y = ps.transform(original_A)

with h5py.File(fileName, 'w') as f:
    f.create_dataset("inputs", data = B)
    f.create_dataset('outputs', data = Y)
    f.create_dataset("inputTimes", data = inputTimes)
    f.create_dataset('outputTimes', data = outputTimes)
    f.create_dataset("originalData", data=np.array(originalDataFrame))
    f.create_dataset('originalInputs',data=original_B)
    f.create_dataset('originalOutputs',data=original_Y)
