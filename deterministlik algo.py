import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

sp500Data = 'data/sp500_data.xlsx'
economicIndicatorData = 'data/credit_data.xlsx'
outputDates = economicIndicatorData[5:(len(economicIndicatorData)-10)] + '_dates.csv'

window = 40
shortMA = 50
longMA = 190
firstPeriod = 1
secondPeriod = 2
thirdPeriod = 3
fourthPeriod = 5
fifthPeriod = 10
initialCapital = 10000

sp500 = pd.read_excel(sp500Data, delimiter=';', index_col='Date', header=0)
economicIndicator = pd.read_excel(economicIndicatorData, delimiter=';', index_col='Date', header=0)
economicIndicator['Difference'] = abs(economicIndicator['Actual'] - economicIndicator['Survey'])
averageDiff = economicIndicator['Difference'].sum()  / len(economicIndicator.index)
economicIndicatorSurprise = economicIndicator.iloc[np.where(economicIndicator['Difference'].values > averageDiff)]

sp500['Short_MA'] = sp500['Close'].rolling(window=shortMA, min_periods=1,center=False).mean().round(2)
sp500['Long_MA'] = sp500['Close'].rolling(window=longMA, min_periods=1,center=False).mean().round(2)
sp500['Trend'] = np.where((sp500['Short_MA'] > sp500['Long_MA']) & (sp500['Close'] > sp500['Short_MA']), 1, 0)
sp500['Trend'] = np.where((sp500['Short_MA'] < sp500['Long_MA']) & (sp500['Close'] < sp500['Short_MA']), -1, sp500['Trend'])
sp500['Signal'] = 0.0

sp500AllObservationDays = sp500.loc[sp500.index.isin(economicIndicatorSurprise.index)]
sp500AllObservationDays['Difference'] = 100 * (sp500AllObservationDays['Close'] - sp500AllObservationDays['Open'])/sp500AllObservationDays['Open']

trendList = [1,0,-1]
countingIndex = 0
averagePositionOpenDays = 0
for surpriseDay in economicIndicatorSurprise.index:
    if countingIndex > window and countingIndex < len(economicIndicatorSurprise)-1:
        sp500ObservationDays = sp500AllObservationDays[countingIndex-window:countingIndex]

        if economicIndicatorSurprise.iloc[countingIndex+1]['Survey'] > economicIndicatorSurprise.iloc[countingIndex+1]['Actual']:

            economicIndicatorPositiveSurprise = economicIndicatorSurprise.iloc[np.where(economicIndicatorSurprise['Actual'].values > economicIndicatorSurprise['Survey'].values)]

            sp500PositiveSurpriseDays = sp500ObservationDays.loc[sp500ObservationDays.index.isin(economicIndicatorPositiveSurprise.index)]

            averageReactionToPositiveSurprisePositiveTrend = sp500PositiveSurpriseDays['Difference'].loc[sp500PositiveSurpriseDays['Trend'] == 1].mean()
            averageReactionToPositiveSurpriseNeutralTrend = sp500PositiveSurpriseDays['Difference'].loc[sp500PositiveSurpriseDays['Trend'] == 0].mean()
            averageReactionToPositiveSurpriseNegativeTrend = sp500PositiveSurpriseDays['Difference'].loc[sp500PositiveSurpriseDays['Trend'] == -1].mean()

            averageReactionToPositiveSurpriseList = [averageReactionToPositiveSurprisePositiveTrend, averageReactionToPositiveSurpriseNeutralTrend, averageReactionToPositiveSurpriseNegativeTrend]
            sp500UnusualReactionToPositiveSurprisePositiveTrend = pd.DataFrame()
            sp500UnusualReactionToPositiveSurpriseNeutralTrend = pd.DataFrame()
            sp500UnusualReactionToPositiveSurpriseNegativeTrend = pd.DataFrame()
            sp500UnusualReactionToPositiveSurpriseList = [sp500UnusualReactionToPositiveSurprisePositiveTrend, sp500UnusualReactionToPositiveSurpriseNeutralTrend, sp500UnusualReactionToPositiveSurpriseNegativeTrend]

            for trend in trendList:
                if averageReactionToPositiveSurpriseList[1-trend] > 0:
                    sp500UnusualReactionToPositiveSurpriseList[1-trend] = sp500UnusualReactionToPositiveSurpriseList[1-trend].append(sp500PositiveSurpriseDays.loc[(sp500PositiveSurpriseDays['Difference'] < 0) & (sp500PositiveSurpriseDays['Trend'] == trend)])
                else:
                    sp500UnusualReactionToPositiveSurpriseList[1-trend] = sp500UnusualReactionToPositiveSurpriseList[1-trend].append(sp500PositiveSurpriseDays.loc[(sp500PositiveSurpriseDays['Difference'] > 0) & (sp500PositiveSurpriseDays['Trend'] == trend)])

            positiveSurpriseDataFrame = pd.DataFrame([[firstPeriod,1,0.0],[firstPeriod,0,0.0],[firstPeriod,-1,0.0],[secondPeriod,1,0.0],
                [secondPeriod,0,0.0],[secondPeriod,-1,0.0],[thirdPeriod,1,0.0],[thirdPeriod,0,0.0],[thirdPeriod,-1,0.0],[fourthPeriod,1,0.0],
                [fourthPeriod,0,0.0],[fourthPeriod,-1,0.0],[fifthPeriod,1,0.0],[fifthPeriod,0,0.0],[fifthPeriod,-1,0.0]],
                                                     columns=['Days', 'Trend', 'Return'])


            for trend in trendList:
                for date in sp500UnusualReactionToPositiveSurpriseList[1-trend].index:
                    for row in positiveSurpriseDataFrame.index:
                        if (positiveSurpriseDataFrame.at[row, 'Days'] == firstPeriod) & (positiveSurpriseDataFrame.at[row, 'Trend'] == trend):
                            positiveSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+firstPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (positiveSurpriseDataFrame.at[row, 'Days'] == secondPeriod) & (positiveSurpriseDataFrame.at[row, 'Trend'] == trend):
                            positiveSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+secondPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (positiveSurpriseDataFrame.at[row, 'Days'] == thirdPeriod) & (positiveSurpriseDataFrame.at[row, 'Trend'] == trend):
                            positiveSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+thirdPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (positiveSurpriseDataFrame.at[row, 'Days'] == fourthPeriod) & (positiveSurpriseDataFrame.at[row, 'Trend'] == trend):
                            positiveSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+fourthPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (positiveSurpriseDataFrame.at[row, 'Days'] == fifthPeriod) & (positiveSurpriseDataFrame.at[row, 'Trend'] == trend):
                            positiveSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+fifthPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                positiveSurpriseDataFrame['Return'].loc[positiveSurpriseDataFrame['Trend'] == trend] = (positiveSurpriseDataFrame['Return'].loc[positiveSurpriseDataFrame['Trend'] == trend]/len(sp500UnusualReactionToPositiveSurpriseList[1-trend])).round(5)

                try:
                    if sp500.iloc[sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1]['Trend'] == trend:
                        daysToCloseThePosition = positiveSurpriseDataFrame.loc[positiveSurpriseDataFrame['Trend'] == trend].loc[abs(positiveSurpriseDataFrame.loc[positiveSurpriseDataFrame['Trend'] == trend]['Return']).idxmax()]['Days']
                        if positiveSurpriseDataFrame.loc[positiveSurpriseDataFrame['Trend'] == trend].loc[abs(positiveSurpriseDataFrame['Return']).idxmax()]['Return'] > 0:
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1),sp500.columns.get_loc('Signal')] += 1
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1+daysToCloseThePosition),sp500.columns.get_loc('Signal')] -= 1
                            averagePositionOpenDays += daysToCloseThePosition
                        else:
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1),sp500.columns.get_loc('Signal')] -= 1
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1+daysToCloseThePosition),sp500.columns.get_loc('Signal')] += 1
                            averagePositionOpenDays += daysToCloseThePosition
                except:
                    pass
            #print(negativeSurpriseDataFrame)

        else:
            economicIndicatorNegativeSurprise = economicIndicatorSurprise.iloc[np.where(economicIndicatorSurprise['Actual'].values < economicIndicatorSurprise['Survey'].values)]
            
            sp500NegativeSurpriseDays = sp500ObservationDays.loc[sp500ObservationDays.index.isin(economicIndicatorNegativeSurprise.index)]

            averageReactionToNegativeSurprisePositiveTrend = sp500NegativeSurpriseDays['Difference'].loc[sp500NegativeSurpriseDays['Trend'] == 1].mean()
            averageReactionToNegativeSurpriseNeutralTrend = sp500NegativeSurpriseDays['Difference'].loc[sp500NegativeSurpriseDays['Trend'] == 0].mean()
            averageReactionToNegativeSurpriseNegativeTrend = sp500NegativeSurpriseDays['Difference'].loc[sp500NegativeSurpriseDays['Trend'] == -1].mean()

            averageReactionToNegativeSurpriseList = [averageReactionToNegativeSurprisePositiveTrend, averageReactionToNegativeSurpriseNeutralTrend, averageReactionToNegativeSurpriseNegativeTrend]
            sp500UnusualReactionToNegativeSurprisePositiveTrend = pd.DataFrame()
            sp500UnusualReactionToNegativeSurpriseNeutralTrend = pd.DataFrame()
            sp500UnusualReactionToNegativeSurpriseNegativeTrend = pd.DataFrame()
            sp500UnusualReactionToNegativeSurpriseList = [sp500UnusualReactionToNegativeSurprisePositiveTrend, sp500UnusualReactionToNegativeSurpriseNeutralTrend, sp500UnusualReactionToNegativeSurpriseNegativeTrend]
            for trend in trendList:
                if averageReactionToNegativeSurpriseList[1-trend] > 0:
                    sp500UnusualReactionToNegativeSurpriseList[1-trend] = sp500UnusualReactionToNegativeSurpriseList[1-trend].append(sp500NegativeSurpriseDays.loc[(sp500NegativeSurpriseDays['Difference'] < 0) & (sp500NegativeSurpriseDays['Trend'] == trend)])
                else:
                    sp500UnusualReactionToNegativeSurpriseList[1-trend] = sp500UnusualReactionToNegativeSurpriseList[1-trend].append(sp500NegativeSurpriseDays.loc[(sp500NegativeSurpriseDays['Difference'] > 0) & (sp500NegativeSurpriseDays['Trend'] == trend)])

            negativeSurpriseDataFrame = pd.DataFrame([[firstPeriod,1,0.0],[firstPeriod,0,0.0],[firstPeriod,-1,0.0],[secondPeriod,1,0.0],
                [secondPeriod,0,0.0],[secondPeriod,-1,0.0],[thirdPeriod,1,0.0],[thirdPeriod,0,0.0],[thirdPeriod,-1,0.0],[fourthPeriod,1,0.0],
                [fourthPeriod,0,0.0],[fourthPeriod,-1,0.0],[fifthPeriod,1,0.0],[fifthPeriod,0,0.0],[fifthPeriod,-1,0.0]],
                columns=['Days', 'Trend', 'Return'])
            
            for trend in trendList:
                for date in sp500UnusualReactionToNegativeSurpriseList[1-trend].index:
                    for row in negativeSurpriseDataFrame.index:
                        if (negativeSurpriseDataFrame.at[row, 'Days'] == firstPeriod) & (negativeSurpriseDataFrame.at[row, 'Trend'] == trend):
                            negativeSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+firstPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (negativeSurpriseDataFrame.at[row, 'Days'] == secondPeriod) & (negativeSurpriseDataFrame.at[row, 'Trend'] == trend):
                            negativeSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+secondPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (negativeSurpriseDataFrame.at[row, 'Days'] == thirdPeriod) & (negativeSurpriseDataFrame.at[row, 'Trend'] == trend):
                            negativeSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+thirdPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (negativeSurpriseDataFrame.at[row, 'Days'] == fourthPeriod) & (negativeSurpriseDataFrame.at[row, 'Trend'] == trend):
                            negativeSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+fourthPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                        elif (negativeSurpriseDataFrame.at[row, 'Days'] == fifthPeriod) & (negativeSurpriseDataFrame.at[row, 'Trend'] == trend):
                            negativeSurpriseDataFrame.at[row, 'Return'] += 100 * (sp500.iloc[sp500.index.get_loc(date)+fifthPeriod]['Close'] - sp500.iloc[sp500.index.get_loc(date)+1]['Open'])/sp500.iloc[sp500.index.get_loc(date)+1]['Open']
                negativeSurpriseDataFrame['Return'].loc[negativeSurpriseDataFrame['Trend'] == trend] = (negativeSurpriseDataFrame['Return'].loc[negativeSurpriseDataFrame['Trend'] == trend]/len(sp500UnusualReactionToNegativeSurpriseList[1-trend])).round(5)

                try:
                    if sp500.iloc[sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1]['Trend'] == trend:
                        daysToCloseThePosition = negativeSurpriseDataFrame.loc[negativeSurpriseDataFrame['Trend'] == trend].loc[abs(negativeSurpriseDataFrame.loc[negativeSurpriseDataFrame['Trend'] == trend]['Return']).idxmax()]['Days']
                        if negativeSurpriseDataFrame.loc[negativeSurpriseDataFrame['Trend'] == trend].loc[abs(negativeSurpriseDataFrame['Return']).idxmax()]['Return'] > 0:
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1),sp500.columns.get_loc('Signal')] += 1
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1+daysToCloseThePosition),sp500.columns.get_loc('Signal')] -= 1
                            averagePositionOpenDays += daysToCloseThePosition
                        else:
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1),sp500.columns.get_loc('Signal')] -= 1
                            sp500.iloc[int(sp500.index.get_loc(economicIndicatorSurprise.index.tolist()[countingIndex+1])+1+daysToCloseThePosition),sp500.columns.get_loc('Signal')] += 1
                            averagePositionOpenDays += daysToCloseThePosition
                except:
                    pass
            #print(negativeSurpriseDataFrame)

    countingIndex +=1

tradeDates = []
portfolio = pd.DataFrame(index=sp500.index)
portfolio['Signal'] = sp500['Signal']
portfolio['Open'] = sp500['Open']
portfolio['Close'] = sp500['Close']

numberOfTrades = len(portfolio.loc[portfolio['Signal'] != 0.0])/2
print('Tehinguid tehti : ', numberOfTrades)

openPositions = 0
shortPositions = 0
averageReturn = 0
for date in portfolio.index:
    if portfolio.at[date, 'Signal'] != 0.0:
        openPositions += 1
        if openPositions % 2 == 1:
            tradeDates.append(date)
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

tradeDatesDataFrame = pd.DataFrame(tradeDates, columns = ['Date'])
tradeDatesDataFrame.to_csv(outputDates)

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
sp500[['Short_MA']].plot(ax=ax1, lw=2.)
sp500[['Long_MA']].plot(ax=ax1, lw=2.)
ax1.plot(sp500.Close[sp500.Signal >= 1.0],
        '^', markersize=10, color='g')
ax1.plot(sp500.Close[sp500.Signal <= -1.0],
        'v', markersize=10, color='k')

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
portfolio['Total'].plot(ax=ax1, lw=2.)
ax1.plot(portfolio.Total[sp500.Signal == 1.0],
         '^', markersize=10, color='g')
ax1.plot(portfolio.Total[sp500.Signal == -1.0],
         'v', markersize=10, color='k')

plt.show()

#print(portfolio.ix[4020:4060])               
#print(sp500.loc[sp500['Signal'] != 0])
#sp500[['Close','SMA']].plot()
#pylab.show()
