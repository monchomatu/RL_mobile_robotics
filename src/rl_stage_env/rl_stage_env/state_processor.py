import numpy as np
from typing import Tuple

class StateProcessor:
    """Process LiDAR data into state representation for DQN"""
    
    def __init__(self, n_lidar_bins: int = 10):
        """
        Args:
            n_lidar_bins: Number of bins to discretize 360° LiDAR scan
        """
        self.n_lidar_bins = n_lidar_bins
        self.max_lidar_range = 3.5  # TurtleBot3 LiDAR max range
        
    def process_lidar(self, scan_data: list) -> np.ndarray:
        """
        Process 360-point LiDAR scan into n bins
        
        Args:
            scan_data: List of distance measurements (360 points)
            
        Returns:
            Array of shape (n_lidar_bins,) with min distances per sector
        """
        if scan_data is None or len(scan_data) == 0:
            return np.ones(self.n_lidar_bins)
        scan_array = np.array(scan_data)
        
        # Replace inf values with max range
        #scan_array[np.isinf(scan_array)] = self.max_lidar_range
        #scan_array[np.isnan(scan_array)] = self.max_lidar_range
        scan_array = np.nan_to_num(scan_array, nan=self.max_lidar_range, 
                               posinf=self.max_lidar_range, 
                               neginf=0.0)

        # Clip values to [0, max_range]
        scan_array = np.clip(scan_array, 0, self.max_lidar_range)
        
        # Divide 360° into bins and take minimum distance in each
        points_per_bin = len(scan_array) // self.n_lidar_bins
        binned_scan = []
        
        for i in range(self.n_lidar_bins):
            start_idx = i * points_per_bin
            end_idx = (i + 1) * points_per_bin if i < self.n_lidar_bins - 1 else len(scan_array)
            bin_min = np.min(scan_array[start_idx:end_idx])
            binned_scan.append(bin_min)
        
        # Normalize to [0, 1]
        return np.array(binned_scan) / self.max_lidar_range
    
    def compute_goal_info(self, 
                         current_pos: Tuple[float, float],
                         goal_pos: Tuple[float, float],
                         current_yaw: float) -> np.ndarray:
        """
        Compute goal distance and relative angle
        
        Args:
            current_pos: (x, y) current position
            goal_pos: (x, y) goal position
            current_yaw: Current heading angle (radians)
            
        Returns:
            Array [distance_to_goal, angle_to_goal] normalized
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        
        # Distance to goal
        distance = np.sqrt(dx**2 + dy**2)
        
        # Angle to goal relative to robot's heading
        angle_to_goal = np.arctan2(dy, dx)
        relative_angle = angle_to_goal - current_yaw
        
        # Normalize angle to [-π, π]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        # Normalize values
        distance_norm = np.clip(distance / 10.0, 0, 1)  # Assume max distance of 10m
        angle_norm = relative_angle / np.pi  # [-1, 1]
        
        return np.array([distance_norm, angle_norm])
    
    def get_state(self,
                  scan_data: list,
                  current_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float],
                  current_yaw: float) -> np.ndarray:
        """
        Combine LiDAR and goal information into complete state
        
        Returns:
            State vector of shape (n_lidar_bins + 2,)
        """
        lidar_state = self.process_lidar(scan_data)
        goal_state = self.compute_goal_info(current_pos, goal_pos, current_yaw)
        
        return np.concatenate([lidar_state, goal_state])