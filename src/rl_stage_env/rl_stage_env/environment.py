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
        
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', 
                                                  self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom',
                                                  self.odom_callback, 10)
        self.state_processor = StateProcessor(n_lidar_bins=10)
        
        # Gazebo service for resetting world
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        
        # State variables
        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.last_position = (0.0, 0.0)
        
        # Goal position (will be randomized)
        self.goal_position = (4.0, 4.0)
        
        # Action space: 5 discrete actions
        self.actions = {
            0: (0.15, 0.0),    # Forward
            1: (0.0, 0.5),     # Rotate left
            2: (0.0, -0.5),    # Rotate right
            3: (0.08, 0.3),    # Forward + left
            4: (0.08, -0.3),   # Forward + right
        }
        
        self.collision_threshold = 0.2  # meters
        self.goal_threshold = 0.3       # meters
        
    def scan_callback(self, msg: LaserScan):
        """Store latest LiDAR scan"""
        self.scan_data = list(msg.ranges)
    
    def odom_callback(self, msg: Odometry):
        """Store latest odometry data"""
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        
        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + 
                        orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + 
                            orientation_q.z * orientation_q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return next state, reward, done
        
        Args:
            action: Action index
            
        Returns:
            next_state, reward, done
        """
        # Execute action
        linear_vel, angular_vel = self.actions[action]
        self.send_velocity(linear_vel, angular_vel)
        
        # Wait for state update
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # Check termination conditions
        done = False
        reward = 0.0
        
        # 1. Check collision
        if self.is_collision():
            reward = -100.0
            done = True
            self.get_logger().info("Collision detected!")
            
        # 2. Check goal reached
        elif self.is_goal_reached():
            reward = 200.0
            done = True
            self.get_logger().info("Goal reached!")
            
        # 3. Ongoing reward
        else:
            reward = self.compute_reward(action)
        
        return self.get_state(), reward, done
    
    def compute_reward(self, action: int) -> float:
        """
        Compute reward for ongoing navigation
        
        Reward components:
        1. Progress toward goal (positive)
        2. Proximity to obstacles (negative)
        3. Action penalty (encourage efficiency)
        """
        # Distance to goal
        current_dist = self.distance_to_goal()
        
        # Progress reward (compare to last position if available)
        if hasattr(self, 'last_distance'):
            progress = self.last_distance - current_dist
            progress_reward = progress * 10.0  # Scale factor
        else:
            progress_reward = 0.0
        
        self.last_distance = current_dist
        
        # Obstacle proximity penalty
        min_obstacle_dist = np.min(self.scan_data) if self.scan_data else 3.5
        if min_obstacle_dist < 0.5:
            obstacle_penalty = -5.0 * (0.5 - min_obstacle_dist)
        else:
            obstacle_penalty = 0.0
        
        # Action penalty (encourage forward motion)
        action_penalty = -0.01 if action in [1, 2] else 0.0  # Penalize pure rotation
        
        # Time penalty (encourage faster completion)
        time_penalty = -0.1
        
        total_reward = progress_reward + obstacle_penalty + action_penalty + time_penalty
        
        return total_reward
    
    def is_collision(self) -> bool:
        """Check if robot has collided with obstacle"""
        if self.scan_data is None:
            return False
        min_distance = np.min(self.scan_data)
        return min_distance < self.collision_threshold
    
    def is_goal_reached(self) -> bool:
        """Check if robot has reached goal"""
        return self.distance_to_goal() < self.goal_threshold
    
    def distance_to_goal(self) -> float:
        """Compute Euclidean distance to goal"""
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def send_velocity(self, linear: float, angular: float):
        """Send velocity command to robot"""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
    
    def reset(self, random_goal: bool = True) -> np.ndarray:
        """
        Reset environment for new episode
        
        Args:
            random_goal: If True, randomize goal position
            
        Returns:
            Initial state
        """
        # Stop robot
        self.send_velocity(0.0, 0.0)
        
        # Reset Gazebo world (resets robot and environment to initial state)
        self.reset_world()
        
        # Randomize goal position
        if random_goal:
            self.goal_position = (
                np.random.uniform(-3.5, 3.5),
                np.random.uniform(-3.5, 3.5)
            )
        
        # Wait for state update after reset
        rclpy.spin_once(self, timeout_sec=0.5)
        
        self.last_distance = self.distance_to_goal()
        
        return self.get_state()
    
    def reset_world(self):
        """Reset Gazebo world using /reset_world service"""
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Reset world service not available')
            return
        
        # Create empty request
        request = Empty.Request()
        
        # Call service
        future = self.reset_world_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            self.get_logger().info('World reset successfully')
        else:
            self.get_logger().error('Failed to reset world')
    
    def get_state(self) -> np.ndarray:
        """Get current state (must be implemented with StateProcessor)"""
        # This will be combined with StateProcessor in the training node
        return self.state_processor.get_state(
            self.scan_data, 
            self.position, 
            self.goal_position, 
            self.yaw
        )