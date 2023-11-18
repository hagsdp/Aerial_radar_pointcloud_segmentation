#### CONFIG & HYPERPARAMETERS ####

# Constants
R = 6378137

# Stationing position, decimal degrees - LAT LON position of the sensor / point of reference
station_lat = 55.680856
station_lon = -4.111961
ini_alt = 20

# Details of the volume to be scanned
max_range = 70 # maximum distance from the sensor to be surveyed
distance_sensor_volume = 5 # minimum distance from the sensor to be surveyed
voxel_size = 3 # granularity of the passes/survey indicated by the voxel size
hor_angle = 56.5 # maximum horizontal angle to be survey (controls azimuthal extent)
ver_angle = 22.5 # maximum vertical angle to be surveyed (controls elevation extent)
bearing = 300 # controls the azimuthal orientation of the experiment (0 to 360 degrees)
pitch_correction = 7 # controls the vertical orientation of the experiment 
