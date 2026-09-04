#!/usr/bin/env python3
"""
RL on Constrained Hardware - Experiment 1
Question: What can you train with 2 ARM cores, 12GB RAM, no GPU?

Progressive environments:
  Level 1: CartPole-v1 (simple, 4 obs, 2 actions)
  Level 2: LunarLander-v3 (medium, 8 obs, 4 actions)
  Level 3: Acrobot-v1 (medium, 6 obs, 3 actions)

For each: train DQN from scratch, measure wall-clock time,
memory usage, episodes to convergence, final performance.
"""

import time
import json
import os
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random


class ReplayBuffer:
    """Simple experience replay buffer."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Simple feedforward DQN."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 batch_size=64, target_update=10, hidden=128):
        self.act_dim = act_dim
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DQN(obs_dim, act_dim, hidden)
        self.target_net = DQN(obs_dim, act_dim, hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.act_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def get_memory_mb():
    """Current process memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def train_environment(env_name, max_episodes=2000, solve_threshold=None,
                      solve_window=100, hidden=128, lr=1e-3):
    """Train DQN on a single environment. Return metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {env_name}")
    print(f"{'='*60}")

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"  Hidden: {hidden}, LR: {lr}")

    agent = DQNAgent(obs_dim, act_dim, lr=lr, hidden=hidden)

    # Metrics
    episode_rewards = []
    episode_losses = []
    start_time = time.time()
    start_mem = get_memory_mb()
    convergence_episode = None
    peak_mem = start_mem

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = []

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        if (episode + 1) % agent.target_update == 0:
            agent.update_target()

        episode_rewards.append(total_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Track memory
        current_mem = get_memory_mb()
        peak_mem = max(peak_mem, current_mem)

        # Check convergence
        if solve_threshold and len(episode_rewards) >= solve_window:
            avg_reward = np.mean(episode_rewards[-solve_window:])
            if avg_reward >= solve_threshold and convergence_episode is None:
                convergence_episode = episode + 1
                print(f"  ✓ SOLVED at episode {convergence_episode} "
                      f"(avg reward: {avg_reward:.1f})")

        # Progress
        if (episode + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            elapsed = time.time() - start_time
            print(f"  Episode {episode+1:5d} | "
                  f"Avg Reward: {avg_r:8.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.0f}s | "
                  f"Mem: {current_mem:.0f}MB")

    elapsed = time.time() - start_time
    env.close()

    # Final stats
    final_avg = np.mean(episode_rewards[-solve_window:])

    results = {
        'env': env_name,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'hidden': hidden,
        'episodes': max_episodes,
        'final_avg_reward': round(float(final_avg), 2),
        'convergence_episode': convergence_episode,
        'wall_clock_seconds': round(elapsed, 1),
        'peak_memory_mb': round(peak_mem, 1),
        'memory_delta_mb': round(peak_mem - start_mem, 1),
        'episodes_per_second': round(max_episodes / elapsed, 2),
        'reward_history': [float(r) for r in episode_rewards],
        'loss_history': [float(l) for l in episode_losses],
    }

    print(f"\n  Results:")
    print(f"    Final avg reward ({solve_window} ep): {final_avg:.1f}")
    print(f"    Convergence: episode {convergence_episode or 'N/A'}")
    print(f"    Wall clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Peak memory: {peak_mem:.0f} MB (delta: {peak_mem-start_mem:.0f} MB)")
    print(f"    Speed: {max_episodes/elapsed:.1f} episodes/sec")

    return results


def main():
    print("="*60)
    print("RL on Constrained Hardware")
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print(f"PyTorch: {torch.__version__}")
    print(f"Initial memory: {get_memory_mb():.0f} MB")
    print("="*60)

    all_results = {}

    # Level 1: CartPole (solved = avg reward >= 475 over 100 episodes)
    all_results['cartpole'] = train_environment(
        'CartPole-v1',
        max_episodes=1000,
        solve_threshold=475,
        hidden=64,
        lr=1e-3,
    )

    # Level 2: Acrobot (solved = avg reward >= -100 over 100 episodes)
    all_results['acrobot'] = train_environment(
        'Acrobot-v1',
        max_episodes=1500,
        solve_threshold=-100,
        hidden=128,
        lr=5e-4,
    )

    # Level 3: LunarLander (solved = avg reward >= 200 over 100 episodes)
    all_results['lunarlander'] = train_environment(
        'LunarLander-v3',
        max_episodes=2000,
        solve_threshold=200,
        hidden=128,
        lr=5e-4,
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Environment':<20} {'Solved?':<10} {'Episodes':<10} "
          f"{'Time':<10} {'Mem (MB)':<10} {'Ep/sec':<10}")
    print("-"*70)

    for name, r in all_results.items():
        solved = f"Ep {r['convergence_episode']}" if r['convergence_episode'] else "No"
        print(f"{r['env']:<20} {solved:<10} {r['episodes']:<10} "
              f"{r['wall_clock_seconds']:<10.0f} {r['peak_memory_mb']:<10.0f} "
              f"{r['episodes_per_second']:<10.1f}")

    # Save full results (without giant histories for the JSON)
    save_results = {}
    for name, r in all_results.items():
        save_r = {k: v for k, v in r.items()
                  if k not in ('reward_history', 'loss_history')}
        # Save sampled histories (every 10th episode)
        save_r['reward_history_sampled'] = r['reward_history'][::10]
        save_r['loss_history_sampled'] = r['loss_history'][::10]
        save_results[name] = save_r

    with open('rl_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print("\nResults saved to rl_results.json")


if __name__ == '__main__':
    main()
