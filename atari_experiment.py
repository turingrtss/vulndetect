#!/usr/bin/env python3
"""
RL on Constrained Hardware - Experiment 2: Finding the Wall
Can DQN learn Pong on 2 ARM cores with no GPU?

Two approaches:
  1. RAM-based (128D observation) - should work, tests if Atari RL is feasible
  2. Image-based (210x160x3 → 84x84x1) - the real test, needs CNN

We measure: wall-clock time per episode, memory, whether it learns at all.
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
import ale_py
from collections import deque
import random

gym.register_envs(ale_py)


class ReplayBuffer:
    def __init__(self, capacity=100000):
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


class DQN_MLP(nn.Module):
    """MLP for RAM observations."""
    def __init__(self, obs_dim, act_dim, hidden=256):
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


class DQN_CNN(nn.Module):
    """CNN for image observations (84x84x1 grayscale)."""
    def __init__(self, act_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        # Calculate conv output size
        dummy = torch.zeros(1, 1, 84, 84)
        conv_out = self.conv(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        # x shape: (batch, 1, 84, 84)
        features = self.conv(x).view(x.size(0), -1)
        return self.fc(features)


def preprocess_frame(frame):
    """Convert 210x160x3 RGB to 84x84 grayscale float."""
    # Grayscale
    gray = np.mean(frame, axis=2).astype(np.float32)
    # Resize to 84x84 using simple slicing (no PIL dependency)
    # Crop to 160x160 from center, then downsample
    gray = gray[18:198, :]  # 180x160
    # Simple 2x2 block average to ~90x80, then crop
    h, w = gray.shape
    new_h, new_w = 84, 84
    row_idx = np.linspace(0, h-1, new_h, dtype=int)
    col_idx = np.linspace(0, w-1, new_w, dtype=int)
    resized = gray[np.ix_(row_idx, col_idx)]
    return resized / 255.0


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def train_atari(mode='ram', max_episodes=500, max_steps_per_ep=5000):
    """Train DQN on Pong. mode='ram' or 'image'."""
    print(f"\n{'='*60}")
    print(f"Atari Pong - {mode.upper()} mode")
    print(f"{'='*60}")

    if mode == 'ram':
        env = gym.make('ALE/Pong-v5', obs_type='ram')
        obs_dim = 128
        net = DQN_MLP(obs_dim, env.action_space.n, hidden=256)
    else:
        env = gym.make('ALE/Pong-v5')
        net = DQN_CNN(env.action_space.n)

    target_net = type(net)(**{k: v for k, v in
        ({'obs_dim': 128, 'act_dim': env.action_space.n, 'hidden': 256} if mode == 'ram'
         else {'act_dim': env.action_space.n}).items()})
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(100000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    eps_end = 0.02
    eps_decay = 0.995
    target_update = 10

    start_time = time.time()
    start_mem = get_memory_mb()
    episode_rewards = []
    ep_times = []

    param_count = sum(p.numel() for p in net.parameters())
    print(f"  Network params: {param_count:,}")
    print(f"  Initial memory: {start_mem:.0f} MB")

    for episode in range(max_episodes):
        ep_start = time.time()
        obs, _ = env.reset()

        if mode == 'image':
            obs = preprocess_frame(obs)

        total_reward = 0
        steps = 0

        while steps < max_steps_per_ep:
            # Select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    if mode == 'ram':
                        state_t = torch.FloatTensor(obs.astype(np.float32) / 255.0).unsqueeze(0)
                    else:
                        state_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
                    action = net(state_t).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if mode == 'image':
                next_obs_proc = preprocess_frame(next_obs)
                buffer.push(obs, action, reward, next_obs_proc, float(done))
                obs = next_obs_proc
            else:
                buffer.push(obs.astype(np.float32) / 255.0, action, reward,
                           next_obs.astype(np.float32) / 255.0, float(done))
                obs = next_obs

            total_reward += reward
            steps += 1

            # Train
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones)

                if mode == 'image':
                    states_t = states_t.unsqueeze(1)
                    next_states_t = next_states_t.unsqueeze(1)

                q_vals = net(states_t).gather(1, actions_t).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(dim=1)[0]
                    targets = rewards_t + gamma * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_vals, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)
        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(net.state_dict())

        ep_time = time.time() - ep_start
        ep_times.append(ep_time)
        episode_rewards.append(total_reward)
        current_mem = get_memory_mb()

        if (episode + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_t = np.mean(ep_times[-10:])
            elapsed = time.time() - start_time
            print(f"  Ep {episode+1:4d} | "
                  f"Avg Reward: {avg_r:6.1f} | "
                  f"Eps: {epsilon:.3f} | "
                  f"Ep Time: {avg_t:.1f}s | "
                  f"Steps: {steps} | "
                  f"Mem: {current_mem:.0f}MB | "
                  f"Elapsed: {elapsed:.0f}s")

    env.close()
    elapsed = time.time() - start_time
    peak_mem = max(get_memory_mb(), current_mem)

    results = {
        'mode': mode,
        'episodes': max_episodes,
        'final_avg_reward': round(float(np.mean(episode_rewards[-50:])), 2),
        'best_avg_reward': round(float(max(
            np.mean(episode_rewards[max(0,i-50):i+1])
            for i in range(49, len(episode_rewards))
        )), 2) if len(episode_rewards) >= 50 else round(float(np.mean(episode_rewards)), 2),
        'wall_clock_seconds': round(elapsed, 1),
        'avg_episode_seconds': round(float(np.mean(ep_times)), 2),
        'peak_memory_mb': round(peak_mem, 1),
        'param_count': param_count,
        'episodes_per_second': round(max_episodes / elapsed, 3),
    }

    print(f"\n  RESULTS ({mode.upper()}):")
    print(f"    Episodes: {max_episodes}")
    print(f"    Final avg reward (50 ep): {results['final_avg_reward']}")
    print(f"    Wall clock: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"    Avg episode time: {np.mean(ep_times):.1f}s")
    print(f"    Peak memory: {peak_mem:.0f} MB")
    print(f"    Throughput: {results['episodes_per_second']:.3f} ep/s")

    return results


def main():
    print("="*60)
    print("RL on Constrained Hardware - Experiment 2: The Atari Wall")
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print(f"PyTorch: {torch.__version__}")
    print(f"Initial memory: {get_memory_mb():.0f} MB")
    print("="*60)

    all_results = {}

    # Phase 1: RAM-based Pong (should be feasible)
    all_results['pong_ram'] = train_atari(mode='ram', max_episodes=500)

    # Phase 2: Image-based Pong (the real test)
    all_results['pong_image'] = train_atari(mode='image', max_episodes=200)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Where's the Wall?")
    print("="*60)

    for name, r in all_results.items():
        print(f"\n  {name}:")
        print(f"    Reward: {r['final_avg_reward']}")
        print(f"    Time: {r['wall_clock_seconds']}s ({r['wall_clock_seconds']/60:.1f} min)")
        print(f"    Speed: {r['episodes_per_second']:.3f} ep/s")
        print(f"    Memory: {r['peak_memory_mb']} MB")
        print(f"    Params: {r['param_count']:,}")

    with open('atari_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to atari_results.json")


if __name__ == '__main__':
    main()
