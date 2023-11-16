import functions as fn
import config
import pandas as pd

# Set the working directory
# Ensure that this path is correct in your config file
path = config.path

# Load dataframes
dji_logs, SBET_file, Oculii_returns = fn.load_dataframes(config.dji_logs_name_first, config.SBET_file_name_first, config.Oculii_returns_name_first)

# Normalize Oculii dataset
df_oculii_norm = fn.normalize_dataset(Oculii_returns)

# Process SBET data
df_sbet = fn.process_SBET(SBET_file, 20, config.times_first, config.station_lat, config.station_lon, config.ASL, config.geodesic_height)

# Process DJI logs
df_dji_logs_time = fn.process_dji_logs(dji_logs, config.filename_first, config.station_lat, config.station_lon, config.feet_to_m)

# Process Oculii data
df_oculii = fn.process_Oculii(df_oculii_norm)

# Merge dataframes
merged_dataframes = fn.merged_dfs(df_oculii, df_sbet, df_dji_logs_time)

# Label dataset
labelled_dataset, label_counts = fn.labelled_dataset(merged_dataframes)

# Output the label counts if needed
print("Targets labelled as dronet:", label_counts)
