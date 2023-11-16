## AUXILIARY FUNCTIONS ###

def transform_coords(lat, lon, dn, de):
    """Transform coordinates with given offsets."""
    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * lat / 180))

    latO = lat + dLat * 180 / math.pi
    lonO = lon + dLon * 180 / math.pi
    return latO, lonO

def round_to_voxel_size(x, size):
    """Round a given value to the closest multiple of the voxel_size."""
    return size * round(x / size)

def create_survey_dataframe(station_lat, station_lon, max_range, voxel_size, hor_angle, ver_angle, bearing, pitch_correction, distance_sensor_volume):
    """Create a DataFrame for the survey mission."""
    # Calculate actual volume sizes
    vertical_FOV = round_to_voxel_size(2 * int(max_range * math.tan(math.radians(ver_angle))), voxel_size)
    horizontal_FOV = round_to_voxel_size(2 * int(max_range * math.tan(math.radians(hor_angle))), voxel_size)
    cube_range = round_to_voxel_size(max_range - distance_sensor_volume, voxel_size)

    # Calculate units to survey
    volume_to_survey = [vertical_FOV / 2, cube_range, horizontal_FOV]
    units_to_survey = [int(i / voxel_size) for i in volume_to_survey]

    # Ensure odd number of rows for zigzag pattern
    if units_to_survey[1] % 2 == 0:
        units_to_survey[1] += 1

    # Calculate initial displacement
    y_arr = units_to_survey[2] * voxel_size / 2
    y_displ = distance_sensor_volume * math.cos(math.radians(90 - bearing)) - y_arr * math.sin(math.radians(90 - bearing))
    x_displ = distance_sensor_volume * math.sin(math.radians(90 - bearing)) + y_arr * math.cos(math.radians(90 - bearing))

    ini_lat, ini_lon = transform_coords(station_lat, station_lon, x_displ, y_displ)

    # Create zigzag pattern
    a = np.arange(units_to_survey[1] * units_to_survey[2]).reshape(units_to_survey[1], units_to_survey[2])
    a[1::2, :] = a[1::2, ::-1]
    b = np.zeros((units_to_survey[0], units_to_survey[1], units_to_survey[2]))

    for i in range(b.shape[0]):
        b[i, :, :] = a if i % 2 == 0 else np.flip(a, (0, 1)) + i * a.size

    # Generate DataFrame
    df = pd.DataFrame([(int(b[i, j, k]), 
                        i * voxel_size + voxel_size / 2 + ini_alt + (j * voxel_size + distance_sensor_volume) * math.sin(math.radians(pitch_correction)), 
                        (j * voxel_size + voxel_size / 2) * math.cos(math.radians(-bearing)) + (k * voxel_size + voxel_size / 2) * math.sin(math.radians(-bearing)), 
                        (k * voxel_size + voxel_size / 2) * math.cos(math.radians(-bearing)) - (j * voxel_size + voxel_size / 2) * math.sin(math.radians(-bearing)))
                       for i in range(b.shape[0]) for j in range(b.shape[1]) for k in range(b.shape[2])],
                      columns=['Pos', 'z', 'x', 'y'])
    df.sort_values(by='Pos', inplace=True)

    # Add LAT LON coordinates
    df['Lat'] = [transform_coords(ini_lat, ini_lon, x, y)[0] for x, y in zip(df['x'], df['y'])]
    df['Lon'] = [transform_coords(ini_lat, ini_lon, x, y)[1] for x, y in zip(df['x'], df['y'])]

    return df

def create_kml_file(df):
    """Convert DataFrame to KML file format."""
    # Process DataFrame for KML format
    df = df.rename(columns={"Pos": "point_name", "z": "height", "Lat": "lat", "Lon": "lon"})
    df = df[['point_name', 'lon', 'lat', 'height']]
    df['heading'] = 180
    df['gimbal'] = -30
    df['speed'] = 6
    df['turnmode'] = 'AUTO'
    df['actions_sequence'] = ''
    df.to_csv('coords.csv', index=False)

    # Convert to KML - This part depends on external script/tool
    # !C:\ProgramData\Anaconda3\python.exe csv2djipilot.py coords.csv -o peke.kml
    return df