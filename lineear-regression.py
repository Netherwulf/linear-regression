import pandas as pd
import quandl
import math

# initialization of data set
df = quandl.get("BSE/BSE", authtoken="7FK44_zsv-ChsbPCWtEF", start_date="2015-04-15")
df = df[['Open', 'High', 'Low', 'Close', 'No. of Trades']]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# features chosen
df = df[['Close', 'HL_PCT', 'PCT_change', 'No. of Trades']]

# forecast column choice
forecast_col = 'Close'

# filling empty data values in data set columns
df.fillna(-99999, inplace=True)

# choice of forecast distance (0.1 means 10% of all rows in data set)
forecast_out = int(math.ceil(0.1 * len(df)))

# initialization of label (Close value from 10% of data forward)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())
