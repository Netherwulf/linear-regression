import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# setting the style of the plot
style.use('ggplot')

# initialization of data set - getting the data from Quandl server
df = quandl.get("FSE/EON_X", authtoken="7FK44_zsv-ChsbPCWtEF", start_date="2003-01-20", end_date='2017-12-28')

# extraction of important columns from the data set
df = df[['Open', 'High', 'Low', 'Close', 'Traded Volume']]

# creation of column containing Highest - Lowest stock price difference in percentage
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0

# creation of column containing Closed - Open stock price difference in percentage
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# features chosen
df = df[['Close', 'HL_PCT', 'PCT_change', 'Traded Volume']]

# forecast column choice
forecast_col = 'Close'

# filling empty data values in data set columns
df.fillna(-99999, inplace=True)

# choice of forecast distance (0.1 means 10% of all rows in data set)
forecast_out = int(math.ceil(0.008 * len(df)))

# initialization of label column (Close value from 0.8% (30 days) of data forward)
df['label'] = df[forecast_col].shift(-forecast_out)

# X - features table
X = np.array(df.drop(['label'], 1))

# usage of preprocessing of table of features
X = preprocessing.scale(X)

# creating features set that will be used to predict labels for each row
X_lately = X[-forecast_out:]

# removing rows which doesn't have label (label column value) from the data set
X = X[:-forecast_out:]

# dropping rows with empty values
df.dropna(inplace=True)

# initialization of label column
y = np.array(df['label'])

# creating training set of records and testing set of records
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # initialization of the classifier
# clf = LinearRegression(n_jobs=-1)

# alternative usage of Support Vector Regression - in this case it has given worse accuracy than linear regression
# clf = svm.SVR()

# # fitting the model to the data
# clf.fit(X_train, y_train)
#
# # saving classifier to pickle
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# loading classifier saved in pickle
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# testing the accuracy of the model
accuracy = clf.score(X_test, y_test)

# making a prediction - forecasting the missing label values for rows without it
forecast_set = clf.predict(X_lately)

# initialization of the new column for predicted values
df['Forecast'] = np.nan

# getting the last date of the data set
last_date = df.iloc[-1].name

# getting the last date in the unix format
last_unix = last_date.timestamp()

# length of day in seconds
one_day = 86400

# getting next date after the last date in unix format
next_unix = last_unix + one_day

# calculating the next dates for predicted values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# creating the plot showing the data set stock close prices and predicted stock close prices
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
