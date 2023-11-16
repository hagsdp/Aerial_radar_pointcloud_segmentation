This section presents an algorithmic approach for data labeling in scenarios where:

- A point-based sensing technique, such as radar or lidar, is employed. The sensor can be either static or dynamic, including ground-based (e.g., automotive) or airborne (e.g., aircraft equipped with a sensor) systems.
- Targets are operating within the sensor's Field of View (FoV).
- In the case of dynamic scenarios, both the targets and ego vehicles (vehicles equipped with the sensor) possess high-frequency and accurate GPS positioning information.

In the demonstrated scenario, two drones are used. One drone is equipped with radar, and the other operates within its FoV. The method involves comparing each radar return with the current position of the target drone to determine whether the return is from the target drone. This process utilizes both the radar data and the high-frequency positional data of the target drone. Detailed theoretical explanations of the implementation are provided in Section 3F of the journal. An excerpt of the results is shown in the picture below

![fig](https://github.com/hagsdp/Aerial_radar_pointcloud_segmentation/assets/35865504/2d94e8b7-dce8-4b50-b45b-ec79bd98a8e3)



