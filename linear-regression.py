import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# initialization of data set
df = quandl.get("FSE/EON_X", authtoken="7FK44_zsv-ChsbPCWtEF", start_date="2003-01-20", end_date='2017-12-28')
df = df[['Open', 'High', 'Low', 'Close', 'Traded Volume']]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# features chosen
df = df[['Close', 'HL_PCT', 'PCT_change', 'Traded Volume']]

# forecast column choice
forecast_col = 'Close'

# filling empty data values in data set columns
df.fillna(-99999, inplace=True)

# choice of forecast distance (0.1 means 10% of all rows in data set)
forecast_out = int(math.ceil(0.008 * len(df)))

# initialization of label (Close value from 10% of data forward)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# X - features table
X = np.array(df.drop(['label'], 1))

# y - labels table
y = np.array(df['label'])

# usage of preprocessing of table of features
X = preprocessing.scale(X)

y = np.array(df['label'])

# creating training set of records and testing set of records
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialization of the classifier
clf = LinearRegression(n_jobs=-1)

# alternative usage of Support Vector Regression
# clf = svm.SVR()

# fitting the model to the data
clf.fit(X_train, y_train)

# testing the accuracy of the model
accuracy = clf.score(X_test, y_test)

print(accuracy)