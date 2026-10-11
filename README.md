# BARN World 037 — ROS 2 Localization & Navigation

A ROS 2 Jazzy navigation system for a four-wheel skid-steer robot in the difficult **BARN World 037** simulation. The system combines local state estimation, visual global correction, LiDAR-based planning and recovery behaviors.

## System overview

```text
wheel odometry ─┐
                ├─ local EKF ── odom → base_link
IMU ────────────┘

ArUco markers ── gradual global correction ── map → odom

LiDAR + map ── Nav2 planner/controller ── cmd_vel ── robot
                       └─ clear / back up / rotate / wait recovery
```

## Highlights

- ROS 2 Jazzy simulation in Gazebo with RViz visualization
- Local EKF using wheel odometry and IMU
- ArUco-based global correction without disruptive pose jumps
- Nav2 with SmacPlanner2D and Regulated Pure Pursuit
- Collision avoidance and custom recovery behavior tree
- Launch and configuration files focused on the hard 037 world

## Run

After building the workspace and sourcing the ROS 2 installation:

```bash
ros2 launch jackal_helper jackal_nav2.launch.py
```

Core topics include `/scan`, `/imu`, `/odometry/local`, `/odometry/aruco` and `/cmd_vel`.

## Project team

University team project by **Aziz Zarrouki, Rayan Brigui and Jihed Khattat**. The `jackal_helper` package names Aziz as maintainer. This repository focuses on the team's package and World 037; the rest of the upstream BARN world corpus is not duplicated here.

Robot meshes and inherited simulation material originate from the ROS ecosystem and remain subject to their original notices and licenses. The package's bundled license is preserved.

## Known limitation

The final report identifies the direct ArUco correction outside the global EKF as an architectural limitation and a candidate for future refinement.
