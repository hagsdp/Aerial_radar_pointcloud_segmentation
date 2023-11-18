## **Mission Planning Algorithm with non-repetitive pattern**

This work contains the development of a mission planning algorithm with a non-repetitive pattern. The primary application in this research is for surveying a sensor's theoretical **Field of View (FoV) volume** for **sensor performance characterization**. This approach is essential to understand the sensor's detection capabilities within each discrete area of its FoV, characterized by voxels.

### **Key Features of the Algorithm**

1. **Volume Discretization in Voxels**: The algorithm discretizes a volume of air into voxels. The centre of the voxels represent waypoints that the drone navigates through.

2. **Parameters for Volume Definition**:

   - **Station coordinates**: 
     - Station Lat
     - Station Lon
     - Station Alt
   - **Volume Extent**: 
     - Distance to sensor
     - Maximum range
     - Azimuth angle
     - Elevation angle
   - **Granularity**:
     - Voxel Size: This parameter allows control over the virtual size of the voxels, enabling a higher or lower level of detail in the survey.
   - **Orientation of the Experiment**:
     - Bearing (ranging from 0 to 360 degrees)
     - Pitch (as an alternative to horizontal surveying)

Below a few examples are provided, considering above parameters which need to be entered in the config.py file, prior to running the script

| Parameter             | Description            | Example Value |
|-----------------------|------------------------|---------------|
| Station_lat           | Latitude of the sensor| (value)       |
| Station_lon           | Longitude of the sensor| (value)      |
| ini_alt               | Altitude of the sensor | 15            |
| distance_sensor_volume| Start of FoV| 30          |
| max_range             | End of FoV          | 120           |
| voxel_size            | Size of Voxels         | 8             |
| hor_angle             | FoV Azimuth       | 120           |
| ver_angle             | FoV Elevation         | 15            |
| bearing               | Bearing Orientation               | 300           |
| pitch_correction      | Pitch Correction       | 0             |

![one](https://github.com/hagsdp/Aerial_radar_pointcloud_segmentation/assets/35865504/ca91388e-944d-41a0-bbd3-1694d9eeb088)

