import pandas as pd
import quandl

df = quandl.get("BSE/BSE", authtoken="7FK44_zsv-ChsbPCWtEF", start_date="2015-04-15")
df = df[['Open', 'High', 'Low', 'Close', 'No. of Trades']]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'No. of Trades']]

print(df.head())
