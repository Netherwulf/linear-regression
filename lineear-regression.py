import pandas as pd
import quandl
import math

df = quandl.get("BSE/BSE", authtoken="7FK44_zsv-ChsbPCWtEF", start_date="2015-04-15")
df = df[['Open', 'High', 'Low', 'Close', 'No. of Trades']]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'No. of Trades']]

forecast_col = 'Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())
