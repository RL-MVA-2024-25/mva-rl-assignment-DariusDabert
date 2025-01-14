
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env_hiv import FastHIVPatient
from tqdm import tqdm
from collections import deque
from evaluate import evaluate_HIV
from pathlib import Path
import random

# Create the environment
ENV = TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)
SAVE_PATH = "DQN_hiv_model"

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
#            nn.Linear(32, 32), nn.ReLU(),
#            nn.Linear(512, 512), nn.ReLU(),
#            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

class ProjectAgent:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 4
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(20000)

        self.q_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def act(self, state, use_random=False):
        if use_random or random.random() < self.epsilon:
            return ENV.action_space.sample()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self):
        checkpoint = torch.load(SAVE_PATH, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.q_network.eval()

def train_and_save_agent():
    agent = ProjectAgent()
    num_episodes = 600
    target_update_frequency = 10

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        observation, _ = ENV.reset()
        total_reward = 0

        for _ in range(200):
            action = agent.act(observation)
            next_observation, reward, done, _, _ = ENV.step(action)
            agent.replay_buffer.push(observation, action, reward, next_observation, done)
            observation = next_observation
            total_reward += reward
            agent.train()

            if done:
                break

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        if episode in np.arange(0,600, 10):
            score = evaluate_HIV(agent, nb_episode=5)
            with open("inter.txt", "a") as file:
                file.write(f"Episode {episode + 1}: Score {score}\n")

    agent.save(SAVE_PATH)
    print(f"Model saved at {SAVE_PATH}")

if __name__ == "__main__":
    train_and_save_agent()