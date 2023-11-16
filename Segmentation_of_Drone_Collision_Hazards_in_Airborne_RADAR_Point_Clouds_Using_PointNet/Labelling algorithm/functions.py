# functions.py

import os
import math
import pandas as pd
import numpy as np
from datetime import timedelta
import config



def transform_coords(station_latitude, station_longitude, reference_latitude, reference_longitude, earth_radius):
    """
    Transforms geographic coordinates from one system to another based on a reference point.

    Parameters:
    - station_latitude (float): Latitude of the station in decimal degrees.
    - station_longitude (float): Longitude of the station in decimal degrees.
    - reference_latitude (float): Latitude of the reference point in decimal degrees.
    - reference_longitude (float): Longitude of the reference point in decimal degrees.
    - earth_radius (float): Radius of the Earth in meters.

    Returns:
    - tuple: A tuple (east_distance, north_distance) representing the East and North distance in meters from the reference point - used to merge with radar local coordinates
    """
    delta_latitude = (reference_latitude - station_latitude) * math.pi / 180
    delta_longitude = (reference_longitude - station_longitude) * math.pi / 180
    
    north_distance = delta_latitude * earth_radius
    east_distance = delta_longitude * (earth_radius * math.cos(math.pi * station_latitude / 180))
    
    return (east_distance, north_distance)


def load_dataframes(dji_logs_name, SBET_file_name, Oculii_returns_name, SBET_sep=" "):
    """
    Loads data from CSV files into Pandas DataFrames.

    Parameters:
    - dji_logs_name (str): Filename of the DJI logs CSV file.
    - SBET_file_name (str): Filename of the SBET CSV file.
    - Oculii_returns_name (str): Filename of the Oculii returns CSV file.
    - SBET_sep (str, optional): Separator used in the SBET file (default is a space).

    Returns:
    - tuple: A tuple containing three Pandas DataFrames corresponding to the files.
    """
    try:
        dji_logs = pd.read_csv(dji_logs_name)
        SBET_file = pd.read_csv(SBET_file_name, sep=SBET_sep)
        Oculii_returns = pd.read_csv(Oculii_returns_name)
    except FileNotFoundError as e:
        
        print(f"File not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
    
    return dji_logs, SBET_file, Oculii_returns


def normalize_dataset(df):
    """
    # The radar hardware comes with two different SDKs one C++ based and one Python based which differ in formatting. 
    # This function standarizes the dataset regardless of the SDK and method used to capture the data
    """
    if df.empty:
        raise ValueError("The dataframe is empty")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    try:
        if 'PTPtime' in df.columns:
            required_columns = ['Timestamp', 'PTPtime', 'Frame Number', 'Version Number',
                                'Num Detections', 'Num Tracks', 'Host Speed', 'Host YawDiff', 
                                'DSPWorkload', 'ARMWorkload', 'Range', 'Doppler', 'Alpha',
                                'Beta', 'Power', 'DotFlag', 'Xpos', 'YPos', 'ZPos']

            df = df[required_columns].rename(columns=lambda x: x.strip())

            new_column_names = ['Host Timestamp', 'PTP Timestamp', 'Frame Number', 'Version Number',
                                'Number Detections', 'Number Tracks', 'Host Speed', 'Host Angle',
                                'DSPWorkload', 'ARMWorkload', 'Range', 'Doppler', 'Alpha', 'Beta',
                                'Power', 'DotFlag', 'Xpos', 'Ypos', 'Zpos']
            df.columns = new_column_names

        elif df['Host Timestamp'].apply(lambda x: len(str(x)) < 12).any():
            df['Host Timestamp'] = df['Host Timestamp'] * 1000

    except KeyError as e:
        raise KeyError(f"Expected column not found: {e}")

    return df




def process_SBET(df_sbet, sampling_rate, times, station_lat, station_lon, ASL, geodesic_height):
    """
    Processes SBET data for coordinate transformation and date-time calculation.

    Parameters:
    - df_sbet (DataFrame): The SBET dataframe to be processed.
    - sampling_rate (int): Sampling rate for downsampling the data.
    - ...

    Returns:
    - DataFrame: Processed SBET dataframe with coordinate transformations and date-time.
    """
    # 'Unnamed: 3', 'Unnamed: 5', and 'Unnamed: 8' are latitude, longitude, and altitude respectively
    df_sbet = df_sbet.iloc[1:, [0, 3, 5, 8]].rename(columns={'Unnamed: 0': 'Index', 'Unnamed: 3': 'Latitude', 'Unnamed: 5': 'Longitude', 'Unnamed: 8': 'Altitude'})
    df_short = df_sbet.iloc[::sampling_rate].reset_index()
    df_short['Latitude'] = df_short['Latitude'] * 180 / 3.14159265
    df_short['Longitude'] = df_short['Longitude'] * 180 / 3.14159265

    # Coordinate transformation using vectorized operations
    df_short[['x', 'y']] = df_short.apply(lambda row: transform_coords(station_lat, station_lon, row['Latitude'], row['Longitude']), axis=1, result_type='expand')
    df_short['z'] = -ASL - geodesic_height + df_short['Altitude']

    # Date-time handling
    start_time = pd.to_datetime(times)
    time_offsets = pd.to_timedelta(df_short.index * 0.1, unit='s')
    df_short['Date-Time'] = start_time + time_offsets

    return df_short



def process_dji_logs(df_dji_logs, filename, station_lat, station_lon, feet_to_m):
    """
    Processes DJI log data with coordinate transformations and date-time calculations.

    Parameters:
    - df_dji_logs (DataFrame): The DJI logs dataframe.
    - filename (str): The filename containing date and time information.
    - station_lat (float): Station latitude.
    - station_lon (float): Station longitude.
    - feet_to_m (float): Conversion factor from feet to meters.

    Returns:
    - DataFrame: Processed DJI logs with additional time and coordinate information.
    """
    # The filename contains a fixed structure, so we pick the required info
    year, month, day, hour, minute, second = int(filename[16:20]), int(filename[21:23]), int(filename[24:26]), int(filename[28:30]), int(filename[31:33]), int(filename[34:36])+0.5
    
    times = [year,month,day,hour, minute, second]

    df_dji_logs = df_dji_logs[['OSD.latitude', 'OSD.longitude', 'OSD.height [ft]', 'OSD.flyTime', 'OSD.yaw [360]']].copy()
    df_dji_logs.columns = df_dji_logs.columns.str.strip()  # Strip whitespace from column names

    # Calculate initial offset in seconds
    initial_offset = pd.to_timedelta(df_dji_logs['OSD.flyTime'].iloc[0])
    start_time = pd.to_datetime(times) + initial_offset

    # Calculate additional time offsets
    additional_offsets = df_dji_logs['OSD.flyTime'].apply(lambda x: pd.to_timedelta(x)).reset_index(drop=True)
    df_dji_logs['Date-Time'] = start_time + additional_offsets - initial_offset

    # Coordinate transformation
    coords = df_dji_logs.apply(lambda row: transform_coords(station_lat, station_lon, row['OSD.latitude'], row['OSD.longitude']), axis=1)
    df_dji_logs[['x_log', 'y_log']] = pd.DataFrame(coords.tolist(), index=df_dji_logs.index)
    df_dji_logs['z_log'] = df_dji_logs['OSD.height [ft]'] * feet_to_m
    
    return df_dji_logs



def process_Oculii(dataset):
    """
    Processes Oculii radar data by filling time gaps and merging time frames.

    Parameters:
    - dataset (DataFrame): The Oculii radar dataset.

    Returns:
    - DataFrame: The processed radar dataset with filled time gaps.
    """
    if dataset.empty:
        raise ValueError("Dataset is empty")

    dataset['Date-Time'] = pd.to_datetime(dataset['Host Timestamp'], unit='ms')

    # Vectorized operation for filling time gaps
    time_diffs = dataset['Date-Time'].diff().dt.total_seconds().fillna(0)
    gaps = (time_diffs > 0.1).nonzero()[0]

    missing_times = []
    for gap in gaps:
        start_time = dataset.loc[gap - 1, 'Date-Time']
        end_time = dataset.loc[gap, 'Date-Time']
        num_missing_entries = int((end_time - start_time).total_seconds() / 0.1)
        missing_times.extend([start_time + timedelta(milliseconds=100 * j) for j in range(1, num_missing_entries)])

    df_missing_times = pd.DataFrame({'Date-Time': missing_times})

    # Merge with existing data
    df_start = pd.merge(df_missing_times, dataset, on='Date-Time', how='outer').sort_values('Date-Time')

    # Fill missing 'Host Timestamp' using forward fill method
    df_start['Host Timestamp'] = df_start['Host Timestamp'].fillna(method='ffill') + 100

    return df_start



def merged_dfs(df_start, df_time, df_dji_logs_time):
    """
    Merges different dataframes and performs yaw rotations.
    This ensures that the radar detections are added in the orientation that the host drone was looking at a given time

    Parameters:
    - df_start (DataFrame): The starting dataframe.
    - df_time (DataFrame): The dataframe containing time data.
    - df_dji_logs_time (DataFrame): The DJI logs dataframe with time data.

    Returns:
    - DataFrame: The final merged dataframe with transformed coordinates.
    """
    df_merged = pd.merge_asof(df_start.sort_values('Date-Time'), df_time, on='Date-Time')
    df_merged_final = pd.merge_asof(df_merged, df_dji_logs_time, on='Date-Time')

    # Vectorized coordinate transformations
    alpha_rad = np.radians(df_merged_final['Alpha'] + df_merged_final['OSD.yaw [360]'])
    beta_rad = np.radians(df_merged_final['Beta'])
    df_merged_final['X_rot'] = df_merged_final['x_log'] + df_merged_final['Range'] * np.sin(alpha_rad) * np.cos(beta_rad)
    df_merged_final['Y_rot'] = df_merged_final['z_log'] - df_merged_final['Range'] * np.sin(beta_rad)
    df_merged_final['Z_rot'] = df_merged_final['y_log'] + df_merged_final['Range'] * np.cos(alpha_rad) * np.cos(beta_rad)

    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'level_0', 'PTP Timestamp', 'Frame Number', 'Version Number', 
                       'Number Detections', 'Number Tracks', 'Host Speed', 'Host Angle', 'DSPWorkload', 
                       'ARMWorkload', 'DotFlag', 'year', 'month', 'day', 'hour', 'minute', 'second', 'OSD.flyTime']
    df_merged_final.drop(columns=[col for col in columns_to_drop if col in df_merged_final.columns], axis=1, inplace=True)
    
    return df_merged_final



def labelled_dataset(df):
    """
    Labels the dataset based on calculated distances and predefined conditions.

    Parameters:
    - df (DataFrame): The dataframe to be labelled.

    Returns:
    - DataFrame: The labelled dataframe.
    """
    # Vectorized distance calculation - calculates the distance of each radar detection at any given time with the target M300 drone. Close/matching detections will be the drone's whereas further away will not.
    df['distance'] = np.sqrt((df['X_rot'] - df['x'])**2 + (df['Z_rot'] - df['y'])**2 + (df['Y_rot'] - df['z'])**2)

    # Label assignment - for any given radar detection that is within 1.5m (plus an additive factor with range) of the real target drone position, the label 'Drone' is awarded
    df['Label'] = np.where(df['distance'] < 1.5 + 0.025 * df['Range'], 2, 1)
    df['Label'] = np.where((df['X_rot'].between(-40, 100)) & (df['Z_rot'].between(-110, -70)), 3, df['Label'])

    label_counts = df['Label'].value_counts()
    
    return df, label_counts

    