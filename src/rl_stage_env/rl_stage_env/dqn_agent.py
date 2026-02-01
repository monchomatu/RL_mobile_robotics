# Base DQNAgent – subject to integration changes

import numpy as np
import random
import pickle
from collections import deque
from copy import deepcopy

from sklearn.neural_network import MLPRegressor


class DQNAgent:
    """
    Deep Q-Network agent (using sklearn MLPRegressor)

    This class is independent of ROS.
    It only handles:
    - Action selection (epsilon-greedy)
    - Experience replay
    - Q-network and target network updates
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        memory_size: int = 2000,
        hidden_layers=(64, 64),
    ):
        # --- Dimensions ---
        self.state_size = state_size
        self.action_size = action_size

        # --- Hyperparameters ---
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # --- Replay Memory ---
        self.memory = deque(maxlen=memory_size)

        # --- Q-Network ---
        self.q_network = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=self.learning_rate,
            max_iter=1,           # Important: train step-by-step
            warm_start=True       # Do not reinitialize weights
        )

        # --- Target Network ---
        self.target_network = deepcopy(self.q_network)

        # Initialize networks with dummy input
        dummy_state = np.zeros((1, self.state_size))
        self.q_network.fit(dummy_state, np.zeros((1, self.action_size)))
        self.target_network.fit(dummy_state, np.zeros((1, self.action_size)))

    # ==========================================================
    # Action Selection
    # ==========================================================

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        state = state.reshape(1, -1)
        q_values = self.q_network.predict(state)
        return int(np.argmax(q_values[0]))

    # ==========================================================
    # Experience Replay
    # ==========================================================

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience in replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    # ==========================================================
    # Training Step
    # ==========================================================

    def train_step(self):
        """
        Sample a minibatch and update Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # Current Q-values
        q_values = self.q_network.predict(states)

        # Target Q-values
        q_next = self.target_network.predict(next_states)
        max_q_next = np.max(q_next, axis=1)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * max_q_next[i]

        # Train Q-network
        self.q_network.fit(states, q_values)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ==========================================================
    # Target Network
    # ==========================================================

    def update_target_network(self):
        """
        Copy weights from Q-network to target network.
        """
        self.target_network = deepcopy(self.q_network)

    # ==========================================================
    # Save / Load
    # ==========================================================

    def save(self, filepath: str):
        """
        Save Q-network to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_network, f)
    
    # --- Alias para compatibilidad con train_node ---
    def act(self, state, training=True):
        # El parámetro 'training' es para que el test_node 
        # pueda desactivar la exploración si lo necesita
        if not training:
            # Si no estamos entrenando, epsilon es 0 (solo la mejor acción)
            temp_epsilon = self.epsilon
            self.epsilon = 0
            action = self.select_action(state)
            self.epsilon = temp_epsilon
            return action
        return self.select_action(state)

    def remember(self, state, action, reward, next_state, done):
        self.store_transition(state, action, reward, next_state, done)

    def replay(self):
        return self.train_step()

    def load(self, filepath: str):
        """
        Load Q-network from disk.
        """
        with open(filepath, 'rb') as f:
            self.q_network = pickle.load(f)

        self.target_network = deepcopy(self.q_network)

