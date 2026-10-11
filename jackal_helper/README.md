# Jackal BARN Challenge

This ROS 2 Jazzy package contains the final localization and navigation system
for the course-specific 037 hard BARN world.

## Build and run

```bash
cd ~/barn_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select jackal_helper
source install/setup.bash
export ROS_DOMAIN_ID=52
ros2 launch jackal_helper jackal_nav2.launch.py headless:=False
```

The final launch uses `037_world_hard.sdf` and the corresponding hard-world map.
Important interfaces are `/scan`, `/imu`, `/odometry/local`,
`/odometry/aruco`, and `/cmd_vel`. The local EKF exclusively publishes
`odom -> base_link`, while `aruco_localization` exclusively publishes
`map -> odom`.

`BARN_runner.launch.py`, `nav2_bringup.launch.py`, and
`gazebo_simulation.py` are inherited benchmark utilities. They do not represent
the final Exercise 6/7 architecture described in the report. The final tested
entry point is `jackal_nav2.launch.py`.

## Authors

- Aziz Zarrouki
- Rayan Brigui
- Jihed Khattat
