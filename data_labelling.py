import numpy as np
import pandas as pd
import math

def label_data(data, months=3, bins=3, trading_days=21):
    
    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Sort the data by ticker and date
    data.sort_values(by=['ticker', 'date'], inplace=True)

    # Compute rolling returns
    period_days = months * trading_days
    data['adjusted_close_months_ago'] = data.groupby('ticker')['adjusted_close'].shift(period_days)
    data['date_months_ago'] = data.groupby('ticker')['date'].shift(period_days)
    data['return'] = (data['adjusted_close'] - data['adjusted_close_months_ago']) / data['adjusted_close_months_ago']

    # Drop rows where any of the new columns is NaN
    data.dropna(subset=['return', 'date_months_ago', 'adjusted_close_months_ago'], inplace=True)
    data = data[['ticker', 'date', 'date_months_ago', 'adjusted_close', 'adjusted_close_months_ago', 'return']]

    # Rank returns for each date and then categorize into bins
    data['rank'] = data.groupby('date')['return'].rank().astype(int)
    max_rank = data['rank'].max()
    bin_size = math.ceil(max_rank / bins)
    data['bin'] = np.ceil(data['rank'] / bin_size).astype(int)

    # Apply one-hot encoding and rename to rankX format
    one_hot = pd.get_dummies(data['bin'], prefix='rank').astype(int)
    data = pd.concat([data, one_hot], axis=1)

    # Drop columns 'rank' and 'bin' as they are no longer needed
    data.drop(columns=['bin'], inplace=True)
    
    return data
