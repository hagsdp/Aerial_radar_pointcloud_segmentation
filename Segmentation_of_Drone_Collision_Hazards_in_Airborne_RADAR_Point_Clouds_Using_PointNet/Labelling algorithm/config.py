# config.py

# Fixed data and configurations - to be modified for each flying mission

# Station coordinates and altitude information
station_lat = 55.68075377
station_lon = -4.1121270
ini_alt = 15
pitch_correction = 0
ASL = 252.2 # site's altitude above sea level
geodesic_height = 54.42
feet_to_m = 0.3048
R = 6378137


# File names 
dji_logs_name = 'DJI_logs_firstpcl.csv'
filename_first = 'DJIFlightRecord_2022-12-12_[12-30-12].txt' # Target Drone DJI logs: required for the starting time
SBET_file_name = 'DJI_20221212122851_0001_sbet.txt' # Target Drone L1 Lidar Logs: Required for target drone high-frequency positioning
Oculii_returns_name = 'pointcloud_1.csv'
times = [2022, 12, 12, 12, 28, 37.22] # Taken from SBET_file_name_first ## TODO: wrap it in a function

# Root directory for flight files
path = "C:\\Users\\Hector\\Desktop\\Work\\Projects\\PointNet aerial\\input\\12122022 - Second flight Str\\"