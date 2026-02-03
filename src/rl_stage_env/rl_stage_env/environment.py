import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from rl_stage_env.state_processor import StateProcessor
import numpy as np
from typing import Tuple
import math


class TurtleBot3Env(Node):
    """ROS2 Environment wrapper for TurtleBot3 navigation"""

    def __init__(self):
        super().__init__('turtlebot3_env')

        # Publishers & Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom/sim', self.odom_callback, 10
        )

        self.state_processor = StateProcessor(n_lidar_bins=10)

        # Gazebo reset service
        self.reset_world_client = self.create_client(Empty, '/reset_sim')

        # Robot state
        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0

        # Goal
        self.goal_position = (4.0, 4.0)
        self.last_distance = None

        # Action space (NO CAMBIADO)
        self.actions = {
            0: (0.08, 0.0),     # Forward
            1: (0.0, 0.6),      # Rotate left
            2: (0.0, -0.6),     # Rotate right
            3: (0.05, 0.4),     # Forward + left
            4: (0.05, -0.4),    # Forward + right
            5: (-0.04, 0.0),    # Backward (penalized)
        }

        self.collision_threshold = 0.2
        self.goal_threshold = 0.3

        # --- Map limits (cave.world) ---
        self.map_limits = {
            'x_min': -7.5,
            'x_max':  7.5,
            'y_min': -7.5,
            'y_max':  7.5,
        }

        # Goal constraints
        self.min_goal_distance = 4.5

    # ======================================================
    # Callbacks
    # ======================================================

    def scan_callback(self, msg: LaserScan):
        self.scan_data = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    # ======================================================
    # Environment API
    # ======================================================

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        linear, angular = self.actions[action]
        self.send_velocity(linear, angular)

        rclpy.spin_once(self, timeout_sec=0.1)

        if self.is_collision():
            return self.get_state(), -100.0, True

        if self.is_goal_reached():
            self.get_logger().info("🎯 GOAL ALCANZADO")
            return self.get_state(), 200.0, True

        reward = self.compute_reward(action)
        return self.get_state(), reward, False

    def reset(self, random_goal: bool = True, episode: int = 0) -> np.ndarray:
        self.send_velocity(0.0, 0.0)
        self.reset_world()

        # Curriculum learning: rango de goal crece con episodios
        goal_range = min(3.5, 1.0 + episode * 0.01)

        if random_goal:
            self.goal_position = self.sample_goal_far_positive(goal_range)

        # Esperar datos válidos
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.scan_data is not None:
                break

        self.last_distance = self.distance_to_goal()
        return self.get_state()

    # ======================================================
    # Goal Sampling
    # ======================================================

    def sample_goal_far_positive(self, goal_range):
        """
        Sample a goal far from the robot, in +x +y direction,
        inside map limits.
        """
        for _ in range(100):
            gx = np.random.uniform(
                max(0.5, -goal_range),
                min(goal_range, self.map_limits['x_max'])
            )
            gy = np.random.uniform(
                max(0.5, -goal_range),
                min(goal_range, self.map_limits['y_max'])
            )

            # Distance from robot (odom origin)
            dist = math.sqrt(gx**2 + gy**2)
            if dist < self.min_goal_distance:
                continue

            # Inside map
            if not (
                self.map_limits['x_min'] <= gx <= self.map_limits['x_max'] and
                self.map_limits['y_min'] <= gy <= self.map_limits['y_max']
            ):
                continue

            return (gx, gy)

        # Safe fallback
        return (goal_range, goal_range)

    # ======================================================
    # Reward & Termination
    # ======================================================

    def compute_reward(self, action: int) -> float:
        current_dist = self.distance_to_goal()

        progress = 0.0
        if self.last_distance is not None:
            progress = self.last_distance - current_dist
        self.last_distance = current_dist

        progress_reward = 15.0 * progress

        obstacle_penalty = 0.0
        min_dist = 3.5

        if self.scan_data:
            valid = [r for r in self.scan_data if np.isfinite(r)]
            if valid:
                min_dist = min(valid)

        if min_dist < 0.25:
            obstacle_penalty = -10.0 * (0.25 - min_dist)

        action_penalty = 0.0
        if action in [1, 2] and min_dist > 0.7:
            action_penalty -= 0.02
        if action == 5:
            action_penalty -= 0.1

        time_penalty = -0.01

        return progress_reward + obstacle_penalty + action_penalty + time_penalty

    def is_collision(self) -> bool:
        if not self.scan_data:
            return False
        valid = [r for r in self.scan_data if np.isfinite(r)]
        return min(valid) < self.collision_threshold if valid else False

    def is_goal_reached(self) -> bool:
        return self.distance_to_goal() < self.goal_threshold

    def distance_to_goal(self) -> float:
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    # ======================================================
    # Helpers
    # ======================================================

    def send_velocity(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    def reset_world(self):
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Reset service not available')
            return

        req = Empty.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def get_state(self) -> np.ndarray:
        return self.state_processor.get_state(
            self.scan_data,
            self.position,
            self.goal_position,
            self.yaw
        )
