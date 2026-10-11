#!/usr/bin/env python3

"""Global robot localization from the known ArUco landmarks in world 037."""

import math
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


MARKER_SIZE = 0.4
CAMERA_X = 0.25

# Marker centres in the Gazebo map frame (metres).
MARKER_POSITIONS = {
    0: (-2.99, 2.00),
    1: (2.99, 2.00),
    2: (0.00, -4.24),
    3: (1.39, -3.75),
    4: (-1.40, -0.81),
    5: (-1.40, 0.81),
}


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class ArucoLocalization(Node):

    def __init__(self):
        super().__init__('aruco_localization')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion = None
        self.odom_xy = None
        self.odom_pose = None
        self.odom_yaw = 0.0
        self.odom_history = deque(maxlen=500)
        self.map_to_odom = None
        self.correction_targets = []
        self.last_target_stamp_ns = None
        self.last_log_time = 0.0
        self.last_reject_log_time = 0.0

        self.declare_parameter('max_marker_distance', 6.0)
        self.declare_parameter('max_reprojection_error', 5.0)
        self.declare_parameter('consensus_distance', 0.45)
        self.declare_parameter('odom_sync_tolerance', 0.15)
        self.declare_parameter('correction_consistency_samples', 5)
        self.declare_parameter('correction_consistency_spread', 0.15)
        self.declare_parameter('correction_max_sigma', 0.40)
        self.declare_parameter('correction_gain', 0.15)
        self.declare_parameter('correction_max_step', 0.02)
        self.declare_parameter('correction_max_total', 3.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.05, self.publish_map_to_odom)

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_1000
        )
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()

        self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.odom_publisher = self.create_publisher(
            Odometry, '/odometry/aruco', 10
        )
        self.debug_publisher = self.create_publisher(
            Image, '/camera/aruco_debug', 10
        )

        self.get_logger().info(
            'ArUco localization started; applying synchronized, '
            f'consistency-checked map->odom corrections from {self.odom_topic}'
        )

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.distortion = np.array(msg.d, dtype=np.float64)

    def odom_callback(self, msg):
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        odom_x = float(msg.pose.pose.position.x)
        odom_y = float(msg.pose.pose.position.y)
        stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
        if self.odom_history and stamp_ns < self.odom_history[-1][0]:
            self.odom_history.clear()
        self.odom_history.append((stamp_ns, odom_x, odom_y, odom_yaw))
        self.odom_xy = (odom_x, odom_y)
        self.odom_yaw = odom_yaw
        if self.map_to_odom is None:
            self.set_map_to_odom(self.initial_x, self.initial_y, self.initial_yaw)
        map_to_odom_yaw = (
            self.map_to_odom[2] if self.map_to_odom is not None else self.initial_yaw
        )
        global_yaw = math.atan2(
            math.sin(map_to_odom_yaw + odom_yaw),
            math.cos(map_to_odom_yaw + odom_yaw),
        )
        self.odom_pose = (
            odom_x,
            odom_y,
            global_yaw,
        )

    def set_map_to_odom(self, robot_x, robot_y, robot_yaw):
        if self.odom_xy is None:
            return
        odom_x, odom_y = self.odom_xy
        self.map_to_odom = self.calculate_map_to_odom(
            robot_x,
            robot_y,
            robot_yaw,
            odom_x,
            odom_y,
            self.odom_yaw,
        )

    def calculate_map_to_odom(
        self,
        robot_x,
        robot_y,
        robot_yaw,
        odom_x,
        odom_y,
        odom_yaw,
    ):
        yaw = math.atan2(
            math.sin(robot_yaw - odom_yaw),
            math.cos(robot_yaw - odom_yaw),
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        x = robot_x - (cosine * odom_x - sine * odom_y)
        y = robot_y - (sine * odom_x + cosine * odom_y)
        return (x, y, yaw)

    def synchronized_odom(self, stamp):
        if not self.odom_history:
            return None
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        sample = min(
            self.odom_history,
            key=lambda item: abs(item[0] - stamp_ns),
        )
        difference = abs(sample[0] - stamp_ns) / 1e9
        tolerance = float(self.get_parameter('odom_sync_tolerance').value)
        if difference > tolerance:
            self.log_rejection(
                'Rejected unsynchronized ArUco frame: '
                f'image/odom difference is {difference:.3f} s'
            )
            return None
        return sample

    def publish_map_to_odom(self):
        if self.map_to_odom is None:
            return
        x, y, yaw = self.map_to_odom
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'odom'
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(transform)

    def reprojection_error(self, detected_corners, rvec, tvec):
        half = MARKER_SIZE / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.distortion,
        )
        observed = np.asarray(detected_corners, dtype=np.float32).reshape(4, 2)
        projected = projected.reshape(4, 2)
        return float(np.mean(np.linalg.norm(observed - projected, axis=1)))

    def log_rejection(self, message):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_reject_log_time >= 1.0:
            self.get_logger().warning(message)
            self.last_reject_log_time = now

    def image_callback(self, msg):
        if self.camera_matrix is None or self.map_to_odom is None:
            return

        odom_sample = self.synchronized_odom(msg.header.stamp)
        if odom_sample is None:
            return
        stamp_ns, odom_x, odom_y, odom_yaw = odom_sample

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.parameters,
        )

        if ids is None:
            self.publish_debug(image, msg)
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_SIZE,
            self.camera_matrix,
            self.distortion,
        )

        estimates = []
        map_to_odom_yaw = self.map_to_odom[2]
        robot_yaw = math.atan2(
            math.sin(map_to_odom_yaw + odom_yaw),
            math.cos(map_to_odom_yaw + odom_yaw),
        )
        max_distance = float(self.get_parameter('max_marker_distance').value)
        max_reprojection = float(
            self.get_parameter('max_reprojection_error').value
        )

        for index, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in MARKER_POSITIONS:
                continue

            tvec = tvecs[index][0]
            distance = float(np.linalg.norm(tvec))
            reprojection = self.reprojection_error(
                corners[index], rvecs[index], tvecs[index]
            )
            if tvec[2] <= 0.0 or distance > max_distance:
                continue
            if reprojection > max_reprojection:
                continue

            # OpenCV camera coordinates: x points right and z points forward.
            relative_x = CAMERA_X + float(tvec[2])
            relative_y = -float(tvec[0])
            cosine = math.cos(robot_yaw)
            sine = math.sin(robot_yaw)
            world_dx = cosine * relative_x - sine * relative_y
            world_dy = sine * relative_x + cosine * relative_y

            marker_x, marker_y = MARKER_POSITIONS[marker_id]
            robot_x = marker_x - world_dx
            robot_y = marker_y - world_dy

            # Measurement uncertainty grows with range and image fit error.
            sigma = min(0.50, max(0.06, 0.05 + 0.02 * distance ** 2 + 0.01 * reprojection))
            estimates.append((robot_x, robot_y, sigma, marker_id))

            cv2.drawFrameAxes(
                image,
                self.camera_matrix,
                self.distortion,
                rvecs[index],
                tvecs[index],
                0.2,
            )

        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        self.publish_debug(image, msg)

        if not estimates:
            return

        # Reject detections that disagree with the median of simultaneous
        # landmarks.  This is robust to one false ID or a poor corner fit.
        median_x = float(np.median([item[0] for item in estimates]))
        median_y = float(np.median([item[1] for item in estimates]))
        consensus_distance = float(
            self.get_parameter('consensus_distance').value
        )
        estimates = [
            item for item in estimates
            if math.hypot(item[0] - median_x, item[1] - median_y)
            <= consensus_distance
        ]
        if not estimates:
            return

        weights = [1.0 / (item[2] ** 2) for item in estimates]
        total_weight = sum(weights)
        robot_x = sum(w * item[0] for w, item in zip(weights, estimates)) / total_weight
        robot_y = sum(w * item[1] for w, item in zip(weights, estimates)) / total_weight

        spread = max(
            [math.hypot(item[0] - robot_x, item[1] - robot_y) for item in estimates]
            or [0.0]
        )
        measurement_sigma = max(
            0.06,
            math.sqrt(1.0 / total_weight),
            spread,
        )
        variance = min(0.25, measurement_sigma ** 2)

        output = Odometry()
        output.header.stamp = msg.header.stamp
        output.header.frame_id = 'map'
        output.child_frame_id = 'base_link'
        output.pose.pose.position.x = robot_x
        output.pose.pose.position.y = robot_y
        output.pose.pose.orientation.z = math.sin(robot_yaw / 2.0)
        output.pose.pose.orientation.w = math.cos(robot_yaw / 2.0)
        output.pose.covariance[0] = variance
        output.pose.covariance[7] = variance
        output.pose.covariance[35] = 0.25
        self.odom_publisher.publish(output)

        max_sigma = float(self.get_parameter('correction_max_sigma').value)
        if measurement_sigma > max_sigma:
            self.correction_targets.clear()
            self.log_rejection(
                f'Rejected uncertain ArUco correction: sigma '
                f'{measurement_sigma:.3f} m exceeds {max_sigma:.3f} m'
            )
            return

        target = self.calculate_map_to_odom(
            robot_x,
            robot_y,
            robot_yaw,
            odom_x,
            odom_y,
            odom_yaw,
        )
        current_x, current_y, current_yaw = self.map_to_odom
        total_correction = math.hypot(
            target[0] - current_x,
            target[1] - current_y,
        )
        max_total = float(self.get_parameter('correction_max_total').value)
        if total_correction > max_total:
            self.correction_targets.clear()
            self.log_rejection(
                f'Rejected ArUco correction of {total_correction:.2f} m '
                f'(maximum {max_total:.2f} m)'
            )
            return

        if (
            self.last_target_stamp_ns is None
            or stamp_ns < self.last_target_stamp_ns
            or (stamp_ns - self.last_target_stamp_ns) / 1e9 > 0.5
        ):
            self.correction_targets.clear()
        self.last_target_stamp_ns = stamp_ns

        required_samples = max(
            3,
            int(self.get_parameter('correction_consistency_samples').value),
        )
        self.correction_targets.append((target[0], target[1]))
        self.correction_targets = self.correction_targets[-required_samples:]
        if len(self.correction_targets) < required_samples:
            return

        target_x = float(np.median([
            item[0] for item in self.correction_targets
        ]))
        target_y = float(np.median([
            item[1] for item in self.correction_targets
        ]))
        target_spread = max(
            math.hypot(item[0] - target_x, item[1] - target_y)
            for item in self.correction_targets
        )
        max_spread = float(
            self.get_parameter('correction_consistency_spread').value
        )
        if target_spread > max_spread:
            self.log_rejection(
                f'Rejected inconsistent ArUco corrections: spread '
                f'{target_spread:.3f} m exceeds {max_spread:.3f} m'
            )
            return

        error_x = target_x - current_x
        error_y = target_y - current_y
        remaining = math.hypot(error_x, error_y)
        gain = float(self.get_parameter('correction_gain').value)
        max_step = float(self.get_parameter('correction_max_step').value)
        step_x = gain * error_x
        step_y = gain * error_y
        step_length = math.hypot(step_x, step_y)
        if step_length > max_step and step_length > 0.0:
            scale = max_step / step_length
            step_x *= scale
            step_y *= scale
            step_length = max_step
        self.map_to_odom = (
            current_x + step_x,
            current_y + step_y,
            current_yaw,
        )

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_log_time >= 1.0:
            marker_ids = [item[3] for item in estimates]
            self.get_logger().info(
                f'Markers {marker_ids}: x={robot_x:.3f}, y={robot_y:.3f}, '
                f'sigma={measurement_sigma:.3f} m, '
                f'correction remaining={remaining:.3f} m, '
                f'step={step_length:.3f} m'
            )
            self.last_log_time = now

    def publish_debug(self, image, source_msg):
        debug_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        debug_msg.header = source_msg.header
        self.debug_publisher.publish(debug_msg)


def main():
    rclpy.init()
    node = ArucoLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
