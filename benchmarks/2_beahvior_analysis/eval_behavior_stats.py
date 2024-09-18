import os
import numpy as np
import pandas as pd
import yaml
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from math import sqrt
from extract_behav import extract_behav
from avg_thi import get_avg_THI

# Calculate correlation metrics
def calculate_metrics(x, y):
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    pearson_corr, p_value = pearsonr(x, y)
    return np.round(r2, 3), np.round(rmse, 3), np.round(pearson_corr, 3), np.round(p_value, 3)

# Function to process data for a specific window size
def process_data_for_behavior(behav_id, window_size, date_list, id_list, daily_THI, pred_behav_dir):
    combined_data = extract_behav(pred_behav_dir, date_list, id_list, window_size, behav_id)

    all_freq_data = []
    all_mean_duration = []
    all_total_duration = []
    all_time_list = []

    for cow_data_dict in combined_data:
        time_list = cow_data_dict['date']
        all_time_list.append(time_list)
        freq_data = cow_data_dict['freq_data']
        mean_duration = cow_data_dict['mean_duration']
        total_duration = cow_data_dict['total_duration']

        all_freq_data.append(freq_data)
        all_mean_duration.append(mean_duration)
        all_total_duration.append(total_duration)

    # Calculate average across all cows
    avg_freq_data = np.mean(all_freq_data, axis=0)
    avg_mean_duration = np.mean(all_mean_duration, axis=0)
    avg_total_duration = np.mean(all_total_duration, axis=0)

    # Ensure lengths match
    if len(avg_freq_data) != len(daily_THI):
        raise ValueError("Length of avg_freq_data and daily_THI do not match.")
    if len(avg_mean_duration) != len(daily_THI):
        raise ValueError("Length of avg_mean_duration and daily_THI do not match.")
    if len(avg_total_duration) != len(daily_THI):
        raise ValueError("Length of avg_total_duration and daily_THI do not match.")

    metrics_freq = calculate_metrics(avg_freq_data, daily_THI)
    metrics_mean_duration = calculate_metrics(avg_mean_duration, daily_THI)
    metrics_total_duration = calculate_metrics(avg_total_duration, daily_THI)

    return {
        'window_size': window_size, 
        'metrics_freq': metrics_freq, 
        'metrics_mean_duration': metrics_mean_duration, 
        'metrics_total_duration': metrics_total_duration,
        'avg_freq_data': avg_freq_data,
        'avg_mean_duration': avg_mean_duration,
        'avg_total_duration': avg_total_duration,
        'time_list': time_list
    }

# ===============================================
""" Main program from here """
if __name__ == '__main__':
    
    date_list = ['0722','0723','0724','0725','0726','0727','0728','0729','0730','0731','0801','0802','0803']
    id_list = range(1, 11)

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--path_dir', type=str, default=os.path.join(current_dir, 'private', "path.yaml")) 
    args = parser.parse_args() 
    
    yaml_dir = args.path_dir

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']
    # pred_behav_dir = file_dirs['pred_behav_dir']

    pred_behav_dir = os.path.join(current_dir, 'pred_behav_data')

    input_dir = os.path.join(sensor_data_dir, 'main_data', 'thi', 'average.csv')
    THI_timestamps, daily_THI = get_avg_THI(input_dir)

    behavior_names = {2: "Standing", 7: "Lying", 3: "Feeding", 6: "Drinking"}
    # window_sizes = {2: 160, 3: 50, 6: 20, 7: 120}  # Specific window sizes for each behavior
    window_sizes = {2: 50, 7: 50, 3: 50, 6: 30}  # Specific window sizes for each behavior

    results = []

    print('Correlating...')

    for behav_id, window_size in window_sizes.items():
        best_metrics = process_data_for_behavior(behav_id, window_size, date_list, id_list, daily_THI, pred_behav_dir)

        results.append({
            "Behavior": behavior_names[behav_id],
            "Window Size": window_size,
            "Frequency Metrics": best_metrics['metrics_freq'],
            "Mean Duration Metrics": best_metrics['metrics_mean_duration'],
            "Total Duration Metrics": best_metrics['metrics_total_duration']
        })


    # Create a DataFrame to display the results summary
    results_df = pd.DataFrame(results)
    # print(results_df)

    # # Formatting the output to match the provided table format
    # formatted_results = []
    # for result in results:
    #     behavior = result['Behavior']
    #     freq_metrics = result['Frequency Metrics']
    #     mean_duration_metrics = result['Mean Duration Metrics']
    #     total_duration_metrics = result['Total Duration Metrics']

    #     formatted_results.append({
    #         "Behavior": behavior,
    #         "Pearson Coefficient": f"[{freq_metrics[2]}, {mean_duration_metrics[2]}, {total_duration_metrics[2]}]",
    #         "P-value": f"[{freq_metrics[3]}, {mean_duration_metrics[3]}, {total_duration_metrics[3]}]",
    #         "R-squared": f"[{freq_metrics[0]}, {mean_duration_metrics[0]}, {total_duration_metrics[0]}]"
    #     })
    
    # formatted_df = pd.DataFrame(formatted_results)
    # print(formatted_df.to_string(index=False))

    # Formatting the output to match the provided table format
    for result in results:
        behavior = result['Behavior']
        freq_metrics = result['Frequency Metrics']
        mean_duration_metrics = result['Mean Duration Metrics']
        total_duration_metrics = result['Total Duration Metrics']

        # print(f"{behavior} [PCC, p-value, R2]:")
        # print(f"\tFreq : [{freq_metrics[2]:.3f}, {freq_metrics[3]:.3f}, {freq_metrics[0]:.3f}]")
        # print(f"\tMean : [{mean_duration_metrics[2]:.3f}, {mean_duration_metrics[3]:.3f}, {mean_duration_metrics[0]:.3f}]")
        # print(f"\tTotal: [{total_duration_metrics[2]:.3f}, {total_duration_metrics[3]:.3f}, {total_duration_metrics[0]:.3f}]")
        print(f"{behavior} [freq, mean, total]:")
        print(f"\tPearson : [{freq_metrics[2]:.3f}, {mean_duration_metrics[2]:.3f}, {total_duration_metrics[2]:.3f}]")
        print(f"\tp-value : [{freq_metrics[3]:.3f}, {mean_duration_metrics[3]:.3f}, {total_duration_metrics[3]:.3f}]")
        print(f"\tR2      : [{freq_metrics[0]:.3f}, {mean_duration_metrics[0]:.3f}, {total_duration_metrics[0]:.3f}]")
        print()


