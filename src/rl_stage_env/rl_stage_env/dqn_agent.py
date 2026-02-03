import numpy as np
import random
import pickle
from collections import deque
from copy import deepcopy
from sklearn.neural_network import MLPRegressor


class DQNAgent:
    """
    Deep Q-Network agent (using sklearn MLPRegressor)
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
        memory_size: int = 10000,
        hidden_layers=(128, 128),
        target_update_freq: int = 100,   # >>> AÑADIDO
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
        self.target_update_freq = target_update_freq  # >>> AÑADIDO

        # --- Step counter ---
        self.step_count = 0  # >>> AÑADIDO

        # --- Replay Memory ---
        self.memory = deque(maxlen=memory_size)

        # --- Q-Network ---
        self.q_network = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=self.learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42
        )

        # --- Target Network ---
        self.target_network = deepcopy(self.q_network)

        # Initialize networks with dummy input (mejor inicialización)
        dummy_X = np.random.randn(1, self.state_size)     # >>> CAMBIO
        dummy_y = np.random.randn(1, self.action_size)    # >>> CAMBIO
        self.q_network.fit(dummy_X, dummy_y)
        self.target_network.fit(dummy_X, dummy_y)

    # ==========================================================
    # Action Selection
    # ==========================================================

    def select_action(self, state: np.ndarray) -> int:
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
        self.memory.append((state, action, reward, next_state, done))

    # ==========================================================
    # Training Step
    # ==========================================================

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # >>> AÑADIDO (loss)

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # Current Q-values
        current_q = self.q_network.predict(states)

        # Target Q-values
        next_q = self.target_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)

        target_q = current_q.copy()

        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # >>> CAMBIO IMPORTANTE: partial_fit
        self.q_network.partial_fit(states, target_q)

        # --- Loss (para logging/debug) ---
        loss = np.mean((target_q - current_q) ** 2)  # >>> AÑADIDO

        # --- Step counter ---
        self.step_count += 1  # >>> AÑADIDO

        # --- Target network update ---
        if self.step_count % self.target_update_freq == 0:  # >>> AÑADIDO
            self.update_target_network()

        # --- Epsilon decay ---
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    # ==========================================================
    # Target Network
    # ==========================================================

    def update_target_network(self):
        self.target_network = deepcopy(self.q_network)

    # ==========================================================
    # Save / Load
    # ==========================================================

    def save(self, filepath: str):
        model_data = {                      # >>> CAMBIO
            'q_network': self.q_network,
            'target_network': self.target_network,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.q_network = model_data['q_network']
        self.target_network = model_data['target_network']
        self.epsilon = model_data['epsilon']
        self.step_count = model_data['step_count']

    # --- Aliases para compatibilidad ---
    def act(self, state, training=True):
        if not training:
            temp_epsilon = self.epsilon
            self.epsilon = 0.0
            action = self.select_action(state)
            self.epsilon = temp_epsilon
            return action
        return self.select_action(state)

    def remember(self, state, action, reward, next_state, done):
        self.store_transition(state, action, reward, next_state, done)

    def replay(self):
        return self.train_step()
