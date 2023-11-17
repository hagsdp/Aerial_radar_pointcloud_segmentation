Mission Planning Algorithm for Sensor Performance Characterization
This repository contains the development of a mission planning algorithm designed for sensor performance characterization. The primary application of this algorithm is in surveying a sensor's theoretical Field of View (FoV) volume using a non-repetitive pattern. This approach is essential to understand the sensor's detection capabilities within each discrete area of its FoV, characterized by voxels.

Key Features of the Algorithm
Volume Discretization in Voxels: The algorithm discretizes a volume of air into voxels. These voxels represent waypoints that the drone navigates through, allowing for a comprehensive survey of the sensor’s FoV.

Parameters for Volume Definition:

Volume Extent:
-Distance to sensor
- Maximum range
- Azimuth angle
- Elevation angle
Granularity:
- Voxel Size: This parameter allows control over the virtual size of the voxels, enabling a higher or lower level of detail in the survey.
Orientation of the Experiment:
- Bearing (ranging from 0 to 360 degrees)
- Pitch (as an alternative to horizontal surveying)

Sensor Station Coordinates: The algorithm considers the latitude and longitude of the station where the sensor is located, enabling precise and context-aware planning of the survey mission.

All these parameters can be entered in the config.py file prior to running the mission
