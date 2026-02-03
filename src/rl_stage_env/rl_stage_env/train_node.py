#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from rl_stage_env.dqn_agent import DQNAgent
from rl_stage_env.environment import TurtleBot3Env
from rl_stage_env.state_processor import StateProcessor
import matplotlib.pyplot as plt
from datetime import datetime
import os

class DQNTrainingNode(Node):
    """Main training node for DQN navigation"""
    
    def __init__(self):
        super().__init__('dqn_training_node')
        
        # Training parameters
        self.n_episodes = 200
        self.max_steps_per_episode = 500
        self.state_size = 12  # 10 LiDAR bins + 2 goal info
        self.action_size = 6
        
        # Initialize components
        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=10)
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64
        )
        
        # Logging
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0
        self.collision_count = 0
        
        # Create results directory
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def get_processed_state(self):
        """Get processed state from environment"""
        return self.state_processor.get_state(
            self.env.scan_data,
            self.env.position,
            self.env.goal_position,
            self.env.yaw
        )
    
    def train(self):
        """Main training loop"""
        self.get_logger().info("Starting DQN training...")
        
        for episode in range(self.n_episodes):
            # Reset environment
            self.env.reset(random_goal=True)
            rclpy.spin_once(self.env, timeout_sec=0.5)
            
            state = self.get_processed_state()
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                # Select and execute action
                action = self.agent.act(state, training=True)
                next_state_raw, reward, done = self.env.step(action)
                next_state = self.get_processed_state()
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                
                # Prevent blocking
                rclpy.spin_once(self.env, timeout_sec=0.01)
            
            # Episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)
            
            if self.env.is_goal_reached():
                self.success_count += 1
            if self.env.is_collision():
                self.collision_count += 1
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.success_count / (episode + 1) * 100
                
                self.get_logger().info(
                    f"Episode: {episode}/{self.n_episodes} | "
                    f"Steps: {step+1} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward (10): {avg_reward:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Success Rate: {success_rate:.1f}%"
                )
            
            # Save model periodically
            if episode % 50 == 0 and episode > 0:
                model_path = os.path.join(self.results_dir, f"model_episode_{episode}.pkl")
                self.agent.save(model_path)
        
        # Final save
        self.agent.save(os.path.join(self.results_dir, "model_final.pkl"))
        self.plot_results()
        
    def plot_results(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Moving average reward
        window = 20
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Reward (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].grid(True)
        
        # Episode steps
        axes[1, 0].plot(self.episode_steps)
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Success rate
        window = 50
        success_history = []
        for i in range(len(self.episode_rewards)):
            if i < window:
                success_history.append(self.success_count / (i + 1))
            else:
                # Count successes in last 'window' episodes
                recent_successes = sum([1 for j in range(i-window+1, i+1) 
                                       if self.episode_rewards[j] > 100])
                success_history.append(recent_successes / window)
        
        axes[1, 1].plot([s * 100 for s in success_history])
        axes[1, 1].set_title(f'Success Rate (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_results.png'))
        self.get_logger().info(f"Results saved to {self.results_dir}")

def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainingNode()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted by user")
    finally:
        trainer.env.send_velocity(0.0, 0.0)
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()