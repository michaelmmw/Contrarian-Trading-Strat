import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Initialisation:
a = pd.read_csv("/Users/michaelwehbe/Desktop/q.csv")
b = pd.read_csv("/Users/michaelwehbe/Desktop/w.csv")
c = pd.read_csv("/Users/michaelwehbe/Desktop/x.csv")

data_temp = pd.concat([a, b, c], axis = 0)



n = len(data_temp['ID'])
n_secs = len(pd.unique(data_temp['ID']))
n_dates = len(pd.unique(data_temp['date']))

#Checking for NAs

if np.sum(data_temp['R'].isnull().sum()) == 0:
    print("No missing Values in the dataset")
else:
    print(np.sum(data_temp['R'].isnull().sum()), "in the dataset")

#Let's now sort the data by entering each security in a column, and each row would represent a date:

data_temp_2 = np.zeros((n_dates, n_secs))


for i in range(0, n_dates):   #This basically does what we described above.
    data_temp_2[i, :] = data_temp.iloc[i*n_secs:n_secs*(i+1), 2]

data = pd.DataFrame(data_temp_2, index= pd.unique(data_temp['date']), columns= pd.unique(data_temp['ID']))

#We now have a dataframe eith rows representing dates and columns representing securities

#Let's create the market index:

market_temp = []
for i in range(0, n_dates):
    market_temp.append(np.mean(data.iloc[i, :]))

market = pd.DataFrame(market_temp, index= pd.unique(data_temp['date']), columns= ['Market_Returns'])

#Let's create the trading signal :

signal_temp = np.zeros((n_dates, n_secs))
for i in range(0, n_dates):
    signal_temp[i, : ] = - (data.iloc[i, :] - market.iloc[i, 0])

signal = pd.DataFrame(signal_temp, index= pd.unique(data_temp['date']), columns= pd.unique(data_temp['ID']))
signal_temp_2 = pd.DataFrame(signal_temp, columns= pd.unique(data_temp['ID']))


#Strategy : Long top decile and short bottom decile
#We will rank order our signals for each date:


sorted_signal = pd.DataFrame(np.sort(signal, axis = 1), index= pd.unique(data_temp['date']))


#We have 690 securities, so 690 returns per date. so the first decile would be the smallest 69 returns and the 10th decile would be the largest 69 returns

#Let's create two matrices, each with the IDs of the securities we are shorting or longing at each time:

long_temp = np.zeros((n_dates, 69))
short_temp = np.zeros((n_dates, 69))
for i in range(0, n_dates):
    long_temp[i, :] = signal_temp_2.sort_values(by = i, axis = 1 ).columns[n_secs-69:n_secs]
    short_temp[i, :] = signal_temp_2.sort_values(by = i, axis = 1 ).columns[0:69]


long_positions = pd.DataFrame(long_temp.astype(int),  index= pd.unique(data_temp['date']))
short_positions = pd.DataFrame(short_temp.astype(int),  index= pd.unique(data_temp['date']))

#We want all the longs to have equal weight in our portfolio, and same for all the shorts
#Hence each long security in the portfolio has weight 1/69 and short has -1/69, which satisfies all the given conditions

#For simplicity of computations let's design a weight matrix:

weights = pd.DataFrame(np.zeros((n_dates, n_secs)), index= pd.unique(data_temp['date']), columns= pd.unique(data_temp['ID']))

for i in range(0, n_dates):
    for j in range(0, 69):
        weights[long_positions.iloc[i, j]][i] = 1/69
        weights[short_positions.iloc[i, j]][i] = -1/69



#Let's now compute the returns of our portfolio:


portfolio_rets_temp = np.array(weights)[:n_dates-1, :]*np.array(data)[1:, :]

portfolio_rets_temp_2 = []
for i in range(0, n_dates -1):
    portfolio_rets_temp_2.append(np.sum(portfolio_rets_temp[i, :]))

portfolio_rets = pd.DataFrame(portfolio_rets_temp_2, columns=['Portfolio Returns'], index= pd.unique(data_temp['date'])[1:])

portfolio_rets.to_excel("ass.xlsx")

#1

#a


market['Dates'] = market.index
portfolio_rets['Dates'] = portfolio_rets.index

d = np.array(market.index)
market.plot(x = 'Dates', y = 'Market_Returns', xticks = d)
plt.title('Market Returns')
plt.xlabel('Time')
plt.ylabel('Market Returns')

portfolio_rets.plot(x = 'Dates', y = 'Portfolio Returns', xticks = d)
plt.title('Portfolio Returns')
plt.xlabel('Time')
plt.ylabel('Portfolio Returns')


market = market.drop('Dates', axis = 1)
portfolio_rets = portfolio_rets.drop('Dates', axis = 1)

#b
#Let's compute the annualized mean return, volatility and sharpe ratio of the strategy and of the market portfolio:

ann_mean_ret_strat = np.mean(portfolio_rets)*252
ann_vol_strat = np.std(portfolio_rets)*np.sqrt(252)
ann_SR_strat = ann_mean_ret_strat/ann_vol_strat  #Since we don't know what the risk free rate is

ann_mean_ret_market = np.mean(market)*252
ann_vol_market = np.std(market)*np.sqrt(252)
ann_SR_market = ann_mean_ret_market/ann_vol_market




tab = pd.DataFrame(np.zeros((3, 2)), columns = ['Strategy', 'Market'], index=['Annualized Mean', 'Annualized volatility', 'Annualized Sharpe Ratio' ])
tab.iloc[0,0] = ann_mean_ret_strat[0]
tab.iloc[1,0] = ann_vol_strat[0]
tab.iloc[2,0] = ann_SR_strat[0]
tab.iloc[0,1] = ann_mean_ret_market[0]
tab.iloc[1,1] = ann_vol_market[0]
tab.iloc[2,0] = ann_SR_market[0]




#c
sns.distplot(market)
plt.xlabel("Returns")
plt.ylabel('Density')
plt.title('Histogram of Market Returns')

#And refer to the returns plot above that indicates stationarity





#d

#To study outliers in market days: We can see from the plot of market returns that there are some extreme values:
#Let's find the top and lower 0.1% of the distribution

lower = np.percentile(market, 1)
upper = np.percentile(market, 99)

higher = []
low = []
for i in range(0, len(market)):
    if market.iloc[i, 0] > upper:
        higher.append(market.index[i])
    elif market.iloc[i, 0] < lower:
        low.append(market.index[i])

higher = pd.DataFrame(higher, columns= ['Upper Market Days Outliers Dates'])
lower = pd.DataFrame(low, columns= ['Lower Market Days Outliers Dates'])
days_outliers = pd.concat([higher, lower], axis = 1)
days_outliers.to_excel('a.xlsx')
#The above are the days outliers


mean_returns_secs =  data.mean(axis = 0)
total_mean = np.mean(data.mean())

lq = np.percentile(mean_returns_secs, 1)
u = np.percentile(mean_returns_secs, 99)



lo = []
hi = []
for i in range(0, len(mean_returns_secs)):
    if mean_returns_secs.iloc[i] > u:
        hi.append(mean_returns_secs.index[i])
    elif mean_returns_secs.iloc[i] < lq:
        lo.append(mean_returns_secs.index[i])


hie = pd.DataFrame(hi, columns= ['Upper Stocks Outliers IDs'])
loe = pd.DataFrame(lo, columns= ['Lower Stocks Outliers IDs'])
stocks_outliers = pd.concat([hie, loe], axis = 1)

#The above are the stocks outliers



#e

np.corrcoef(market.iloc[1:, 0], portfolio_rets.iloc[:, 0])

#It is only dollar neutral


#f


weights_long = pd.DataFrame(np.zeros((n_dates, n_secs)), index= pd.unique(data_temp['date']), columns= pd.unique(data_temp['ID']))
weights_short = pd.DataFrame(np.zeros((n_dates, n_secs)), index= pd.unique(data_temp['date']), columns= pd.unique(data_temp['ID']))

for i in range(0, n_dates):
    for j in range(0, 69):
        weights_long[long_positions.iloc[i, j]][i] = 1/69
        weights_short[short_positions.iloc[i, j]][i] = -1/69



#Let's now compute the returns of our portfolio:


portfolio_rets_temp_long = np.array(weights_long)[:n_dates-1, :]*np.array(data)[1:, :]
portfolio_rets_temp_short = np.array(weights_short)[:n_dates-1, :]*np.array(data)[1:, :]

portfolio_rets_temp_2_long = []
portfolio_rets_temp_2_short = []
for i in range(0, n_dates -1):
    portfolio_rets_temp_2_long.append(np.sum(portfolio_rets_temp_long[i, :]))
    portfolio_rets_temp_2_short.append(np.sum(portfolio_rets_temp_short[i, :]))

np.corrcoef(portfolio_rets_temp_2_long, portfolio_rets_temp_2_short)





#2

def lagged_signal(lag):
    signal_temp_lag = np.zeros((n_dates, n_secs))
    for e in range(lag - 1, n_dates):
        signal_temp_lag[e, :] = - (data.iloc[e - lag + 1, :] - market.iloc[e - lag + 1, 0])
    signal_lag = pd.DataFrame(signal_temp_lag, index=pd.unique(data_temp['date']), columns=pd.unique(data_temp['ID']))
    return signal_lag, signal_temp_lag


def lagged_strat(lag):

    signal_lag, signal_temp_lag = lagged_signal(lag)
    signal_temp_lag_2 = pd.DataFrame(signal_temp_lag, columns= pd.unique(data_temp['ID']))


    #Strategy : Long top decile and short bottom decile
    #We will rank order our signals for each date:


    sorted_signal_lag = pd.DataFrame(np.sort(signal_lag, axis = 1), index= pd.unique(data_temp['date']))


    #We have 690 securities, so 690 returns per date. so the first decile would be the smallest 69 returns and the 10th decile would be the largest 69 returns

    #Let's create two matrices, each with the IDs of the securities we are shorting or longing at each time:

    long_temp_lag = np.zeros((n_dates, 69))
    short_temp_lag = np.zeros((n_dates, 69))
    for i in range(0, n_dates):
        long_temp_lag[i, :] = signal_temp_lag_2.sort_values(by = i, axis = 1 ).columns[n_secs-69:n_secs]
        short_temp_lag[i, :] = signal_temp_lag_2.sort_values(by = i, axis = 1 ).columns[0:69]


    long_positions_lag = pd.DataFrame(long_temp_lag.astype(int),  index= pd.unique(data_temp['date']))
    short_positions_lag = pd.DataFrame(short_temp_lag.astype(int),  index= pd.unique(data_temp['date']))

    #We want all the longs to have equal weight in our portfolio, and same for all the shorts
    #Hence each long security in the portfolio has weight 1/69 and short has -1/69, which satisfies all the given conditions

    #For simplicity of computations let's design a weight matrix:

    weights_lag = pd.DataFrame(np.zeros((n_dates, n_secs)), index= pd.unique(data_temp['date']), columns= pd.unique(data_temp['ID']))

    for i in range(0, n_dates):
        for j in range(0, 69):
            weights_lag[long_positions_lag.iloc[i, j]][i] = 1/69
            weights_lag[short_positions_lag.iloc[i, j]][i] = -1/69


    #Let's now compute the returns of our portfolio:


    portfolio_rets_temp_lag = np.array(weights_lag)[:n_dates-1, :]*np.array(data)[1:, :]

    portfolio_rets_temp_lag_2 = []
    for i in range(0, n_dates -1):
        portfolio_rets_temp_lag_2.append(np.sum(portfolio_rets_temp_lag[i, :]))

    portfolio_rets_lag = pd.DataFrame(portfolio_rets_temp_lag_2, columns=['Portfolio Returns'], index= pd.unique(data_temp['date'])[1:])


    #a
    #Let's compute the annualized mean return, volatility and sharpe ratio of the strategy and of the market portfolio:

    ann_mean_ret_strat_lag = np.mean(portfolio_rets_lag)*252
    ann_vol_strat_lag = np.std(portfolio_rets_lag)*np.sqrt(252)
    ann_SR_strat_lag = ann_mean_ret_strat_lag/ann_vol_strat_lag  #Since we don't know what the risk free rate is

    return float(ann_mean_ret_strat_lag), float(ann_vol_strat_lag), float(ann_SR_strat_lag)





table = pd.DataFrame(np.zeros((5, 3)), columns= ['Annualized Mean Return', 'Annualized Volatility', 'Annualized Sharpe Ratio'], index=['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Lag 5'])
for i in range(1, 6):
    print(i)
    table.iloc[i - 1,0], table.iloc[i - 1,1], table.iloc[i - 1,2] = lagged_strat(i)

table.to_excel('table.xlsx')




#b

flat_file = pd.DataFrame(data = np.zeros((1, 6)), columns= ['pid', 'd', 'id', 'k', 'w', 'vid'])

flat_file['pid'] = 928388594
flat_file['vid'] = 0


def weight_matrix(lag):
    signal_lag, signal_temp_lag = lagged_signal(lag)
    signal_temp_lag_2 = pd.DataFrame(signal_temp_lag, columns= pd.unique(data_temp['ID']))


    #Strategy : Long top decile and short bottom decile
    #We will rank order our signals for each date:


    #sorted_signal_lag = pd.DataFrame(np.sort(signal_lag, axis = 1), index= pd.unique(data_temp['date']))


    #We have 690 securities, so 690 returns per date. so the first decile would be the smallest 69 returns and the 10th decile would be the largest 69 returns

    #Let's create two matrices, each with the IDs of the securities we are shorting or longing at each time:

    long_temp_lag = np.zeros((n_dates, 69))
    short_temp_lag = np.zeros((n_dates, 69))
    for i in range(0, n_dates):
        long_temp_lag[i, :] = signal_temp_lag_2.sort_values(by = i, axis = 1 ).columns[n_secs-69:n_secs]
        short_temp_lag[i, :] = signal_temp_lag_2.sort_values(by = i, axis = 1 ).columns[0:69]


    long_positions_lag = pd.DataFrame(long_temp_lag.astype(int),  index= pd.unique(data_temp['date']))
    short_positions_lag = pd.DataFrame(short_temp_lag.astype(int),  index= pd.unique(data_temp['date']))

    return long_positions_lag, short_positions_lag



long_lag_one, short_lag_one = weight_matrix(1)
long_lag_two, short_lag_two = weight_matrix(2)
long_lag_three, short_lag_three = weight_matrix(3)
long_lag_four, short_lag_four = weight_matrix(4)
long_lag_five, short_lag_five = weight_matrix(5)

n_dates
n_secs


RES = pd.concat([long_lag_one, long_lag_two, long_lag_three, long_lag_four, long_lag_five, short_lag_one, short_lag_two, short_lag_three, short_lag_four, short_lag_five], axis = 1)
res = RES.transpose()
res.shape

#3028 dates, each date has to be repeated 690 times
#690 weights, 345 long and 345 short

DF = np.zeros((690*3028, 6))
j = 0
for i in range(690, 690*3028 + 1, 690):
    #if j <= 690:
    DF[i - 690 : i, 2] = res.iloc[:, j]
    #DF.iloc[i - 690: i, 1] = res.columns[j]
    j += 1



DF[0:1044660, 4] = 1/69
DF[1044661:690*3028, 4] = -1/69
DF[:, 0] = 928388594

k = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
j = 0
for i in range(69*3028, 690*3028 + 1, 69*3028):
    DF[i - 69*3028:i, 3] = k[j]
    j+= 1

DF_v = pd.DataFrame(DF, columns= ['pid', 'd', 'id', 'k', 'w', 'vid'])

j = 0
for i in range(690, 690*3028 + 1, 690):
    DF_v.iloc[i - 690: i, 1] = res.columns[j]
    j += 1



DF_v.to_csv('upload_Michael_Wehbe_project_B.csv')


s = '2+2+3*456456*5645/23423'

a = ['+', '-', '*', '/']
v = []
m = 0
operators = []
for i, j in enumerate(s):
    if j in a:
        v.append(int(s[m:i]))
        operators.append(j)
        m = (i+1)
v.append(int(s[m:int(len(s))]))


