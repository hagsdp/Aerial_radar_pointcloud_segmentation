#### MAIN MISSION PLANNING ####

import config
import functions as fn

df = fn.create_survey_dataframe(
    config.station_lat, 
    config.station_lon, 
    config.ini_alt,
    config.max_range, 
    config.voxel_size, 
    config.hor_angle, 
    config.ver_angle, 
    config.bearing, 
    config.pitch_correction, 
    config.distance_sensor_volume)


fn.create_kml_file(df)
