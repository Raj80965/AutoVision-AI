"""
Deep Q-Network (DQN) Reinforcement Learning Agent
Upgrade from tabular Q-Learning to Neural Network based DQN using PyTorch.

Architecture:
  Input:  5 continuous state features
  Hidden: 128 → 64 neurons (ReLU)
  Output: 4 Q-values (one per action)

Features:
  - Experience Replay Buffer (10,000 transitions)
  - Target Network for stable training
  - Epsilon-greedy exploration with decay
  - Backward-compatible API (same get_action / update / save / load)
"""

import random
import os
import math
import pickle
from collections import deque

import numpy as np

# ── PyTorch (optional — fallback to Q-Table if not installed) ──────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found — falling back to Q-Table RL agent.")


# ══════════════════════════════════════════════════════════════════════════════
#  Neural Network Definition
# ══════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class DQNNet(nn.Module):
        """
        Fully-connected DQN:
          state(5) → 128 → 64 → Q-values(4)
        """
        def __init__(self, state_size=5, action_size=4):
            super(DQNNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )

        def forward(self, x):
            return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Experience Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done=False):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
#  DQN Agent  (PyTorch path)
# ══════════════════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Deep Q-Network agent.
    Uses experience replay + target network for stable training.
    """
    STATE_SIZE  = 5   # [speed_norm, lane_pos, has_obstacle, ttc_norm, lane_deviation]
    ACTION_SIZE = 4   # throttle / brake / steer-left / steer-right

    def __init__(self,
                 learning_rate=1e-3,
                 discount=0.95,
                 epsilon=0.5,
                 epsilon_min=0.05,
                 epsilon_decay=0.998,
                 batch_size=64,
                 target_update_freq=200):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (trained every step)
        self.policy_net = DQNNet(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        # Target network (updated every N steps — stabilises training)
        self.target_net = DQNNet(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn    = nn.MSELoss()
        self.replay     = ReplayBuffer(10_000)

        self.discount           = discount
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self.total_updates   = 0
        self.episode_reward  = 0.0
        self._prev_state_vec = None

        print(f"[DQN] Initialized on device: {self.device}")

    # ── State encoding ──────────────────────────────────────────────────────
    def _encode_state(self, speed, lane_pos, has_obstacle, ttc, lane_deviation=0.0):
        """Convert raw values to normalised float vector."""
        return np.array([
            min(speed / 80.0, 1.0),          # normalised speed
            float(lane_pos),                  # 0..1
            float(has_obstacle),              # 0 or 1
            min(ttc / 10.0, 1.0),            # normalised TTC
            float(np.clip(lane_deviation + 0.5, 0.0, 1.0))   # deviation centred
        ], dtype=np.float32)

    # ── Public API — same as old Q-table agent ───────────────────────────────
    def get_state_key(self, speed, lane_pos, has_obstacle, ttc, lane_deviation=0.0):
        """Returns state vector (replaces old tuple key)."""
        return self._encode_state(speed, lane_pos, has_obstacle, ttc, lane_deviation)

    def get_action(self, state_vec):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.ACTION_SIZE - 1)

        with torch.no_grad():
            t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(t)
            return int(q_vals.argmax().item())

    def update(self, state_vec, action, reward, next_state_vec, done=False):
        """Store transition and train one step if buffer is ready."""
        self.replay.push(state_vec, action, reward, next_state_vec, done)
        self.episode_reward += reward

        if len(self.replay) < self.batch_size:
            return  # not enough data yet

        self._train_step()
        self.total_updates += 1

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Sync target network periodically
        if self.total_updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self):
        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values (Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.discount * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def calculate_reward(self, speed, target_speed, collision, ttc,
                         lane_center, obstacle_avoided):
        """Reward function — same logic as before, slightly tuned."""
        reward = 0.0

        if not collision:
            speed_diff = abs(speed - target_speed)
            reward += 10.0 * (1.0 - min(speed_diff / max(target_speed, 1), 1.0))

        lane_error = abs(lane_center - 0.5)
        reward += 5.0 * (1.0 - min(lane_error * 2.0, 1.0))

        if ttc > 3.0:
            reward += 3.0
        elif 0 < ttc < 1.5:
            reward -= 20.0

        if obstacle_avoided:
            reward += 30.0

        if collision:
            reward -= 100.0

        return reward

    def save(self, filename="dqn_model.pth"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon':    self.epsilon,
            'total_updates': self.total_updates
        }, filename)
        print(f"[SAVED] DQN model saved -> {filename}  "
              f"(updates={self.total_updates}, eps={self.epsilon:.4f})")

    def load(self, filename="dqn_model.pth"):
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.epsilon       = ckpt.get('epsilon', self.epsilon_min)
            self.total_updates = ckpt.get('total_updates', 0)
            print(f"[OK] DQN model loaded <- {filename}  "
                  f"(eps={self.epsilon:.4f}, updates={self.total_updates})")
        else:
            print("[WARN] No DQN model found - starting from scratch.")


# ══════════════════════════════════════════════════════════════════════════════
#  Fallback — Q-Table agent (used when PyTorch not installed)
# ══════════════════════════════════════════════════════════════════════════════
class QTableAgent:
    """Original tabular Q-Learning kept as fallback."""
    def __init__(self, learning_rate=0.1, discount=0.95, epsilon=0.2,
                 epsilon_min=0.01, epsilon_decay=0.995):
        from collections import defaultdict
        self.q_table        = defaultdict(float)
        self.learning_rate  = learning_rate
        self.discount       = discount
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.episode_reward = 0.0
        self.total_updates  = 0

    def get_state_key(self, speed, lane_pos, has_obstacle, ttc, lane_deviation=0.0):
        return (int(speed / 10), int(lane_pos * 5), int(has_obstacle), int(min(ttc, 5)))

    def get_action(self, state_key):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        q_values = [self.q_table[(state_key, a)] for a in range(4)]
        return q_values.index(max(q_values)) if any(q_values) else 0

    def update(self, state_key, action, reward, next_state_key, done=False):
        old  = self.q_table[(state_key, action)]
        nmax = max(self.q_table[(next_state_key, a)] for a in range(4))
        self.q_table[(state_key, action)] = old + self.learning_rate * (
            reward + self.discount * nmax - old)
        self.episode_reward += reward
        self.total_updates  += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, speed, target_speed, collision, ttc,
                         lane_center, obstacle_avoided):
        reward = 0.0
        if not collision:
            speed_diff = abs(speed - target_speed)
            reward += 10.0 * (1.0 - min(speed_diff / max(target_speed, 1), 1.0))
        lane_error = abs(lane_center - 0.5)
        reward += 5.0 * (1.0 - min(lane_error * 2.0, 1.0))
        if ttc > 3.0:   reward += 3.0
        elif 0 < ttc < 1.5: reward -= 20.0
        if obstacle_avoided: reward += 30.0
        if collision:   reward -= 100.0
        return reward

    def save(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"[SAVED] Q-table saved ({len(self.q_table)} states, eps={self.epsilon:.4f})")

    def load(self, filename="q_table.pkl"):
        from collections import defaultdict
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded = pickle.load(f)
            self.q_table = defaultdict(float, loaded)
            print(f"[OK] Loaded Q-table with {len(self.q_table)} states.")
        else:
            print("[WARN] No existing Q-table found. Starting fresh.")


# ══════════════════════════════════════════════════════════════════════════════
#  Public export — RLAgent auto-selects DQN or Q-Table
# ══════════════════════════════════════════════════════════════════════════════
def RLAgent(**kwargs):
    """
    Factory function — returns DQNAgent if PyTorch available, else QTableAgent.
    advanced_drive.py uses this exactly like before: agent = RLAgent()
    """
    if TORCH_AVAILABLE:
        return DQNAgent(**kwargs)
    else:
        # Strip DQN-only kwargs
        safe_kwargs = {k: v for k, v in kwargs.items()
                       if k in ('learning_rate', 'discount', 'epsilon',
                                 'epsilon_min', 'epsilon_decay')}
        return QTableAgent(**safe_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  DQN RL Agent — Self Test")
    print("=" * 55)

    agent = RLAgent()
    backend = "DQN (PyTorch)" if TORCH_AVAILABLE else "Q-Table (fallback)"
    print(f"Backend: {backend}")

    state = agent.get_state_key(speed=30, lane_pos=0.5,
                                 has_obstacle=True, ttc=2.0)
    print(f"State vector: {state}")

    action = agent.get_action(state)
    names = ["Throttle", "Brake", "Steer Left", "Steer Right"]
    print(f"Action: {names[action]}")

    reward = agent.calculate_reward(
        speed=35, target_speed=40, collision=False,
        ttc=2.5, lane_center=0.48, obstacle_avoided=False)
    print(f"Reward: {reward:.2f}")

    next_state = agent.get_state_key(speed=38, lane_pos=0.52,
                                      has_obstacle=False, ttc=6.0)
    # Fill buffer with dummy data so training step fires
    for _ in range(70):
        agent.update(state, action, reward, next_state)

    print(f"Updates: {agent.total_updates}  Epsilon: {agent.epsilon:.4f}")

    agent.save()
    agent.load()

    print("=" * 55)
    print("✅ DQN Agent test passed!")
    print("=" * 55)