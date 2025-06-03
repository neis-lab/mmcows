
import numpy as np
import pandas as pd

def get_avg_THI(input_dir):

    # Plot indoor THI
    data = pd.read_csv(input_dir).values # skip the firt row, otherwise: header = True
    data = np.hstack((data[:,0].reshape((-1,1)),data[:,3].reshape((-1,1))))

    # Convert numpy array to pandas DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'value'])

    # Convert Unix timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Set the timestamp column as the index
    df.set_index('timestamp', inplace=True)

    # Group by day and calculate the percentage of 1 values
    df.index = df.index.tz_localize('UTC').tz_convert('America/Chicago')
    daily_percentage = df.groupby(df.index.date)['value'].mean()

    # Get representative timestamps for each day
    representative_timestamps = pd.to_datetime(daily_percentage.index)

    # print(representative_timestamps)

    return representative_timestamps[1:-1], daily_percentage.values[1:-1]

