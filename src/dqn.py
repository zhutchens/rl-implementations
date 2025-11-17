import torch
import torch.nn as nn
import torch.optim as opt
from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, max_length: int, batch_size: int):
        self.buffer = deque(maxlen = max_length)
        self.batch_size = batch_size

    def store(self, transition: tuple):
        self.buffer.append(transition)

    def get_batch(self):
        return random.sample(self.buffer, k = self.batch_size)

class DQN(nn.Module):
    '''
    DQN agent with CNN layers for learning from image-based tasks such as Atari games.
    '''
    def __init__(
            self, 
            lr: float, 
            gamma: float, 
            in_feats: int,
            hidden_size: int,
            out_feats: int,
        ):
        super(DQN, self).__init__()
        self.lr = lr
        self.gamma = gamma

        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        
        self.forward_layers = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_feats)
        )
        self.optimizer = opt.AdamW(self.parameters(), lr)

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.forward_layers(state)

    def save_model(self, path: str) -> None:
        torch.save({
            'weights': self.state_dict(),
            'optimizer_weights': self.optimizer.state_dict(),
            'params': {
                'lr': self.lr,
                'gamma': self.gamma,
                'in_feats': self.in_feats,
                'hidden_size': self.hidden_size,
                'out_feats': self.out_feats,
            }
        }, path)

    @staticmethod
    def load_model(path: str, device: str) -> 'DQN':
        checkpoint = torch.load(path, map_location = torch.device(device))
        params = checkpoint['params']
        dqn = DQN(**params).to(device)
        dqn.load_state_dict(checkpoint['weights'])
        dqn.optimizer.load_state_dict(checkpoint['optimizer_weights'])
        return dqn 
    

class Agent:
    '''
    Double dqn agent. Does not work with image-based environments
    '''
    def __init__(self, policy_net: DQN, target_net: DQN, device: str, decay_rate: float = 0.99, max_buffer_length: int = 100000, batch_size: int = 64):
        self.policy_net = policy_net
        self.target_net = target_net
        self.epsilon = 1
        self.decay_rate = decay_rate
        self.device = device

        self.max_buffer_length = max_buffer_length
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_buffer_length, batch_size)

    def update_policy(self):
        transitions = self.memory.get_batch()
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.stack(states)
        actions = torch.as_tensor(actions).to(self.device)
        rewards = torch.as_tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.as_tensor(dones).to(self.device)

        current_q_values = torch.gather(self.policy_net.forward(states), 1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_best_actions = torch.argmax(self.policy_net.forward(next_states), dim = 1)
            next_target_q_values = self.target_net.forward(next_states)
            max_next_q_values = torch.gather(next_target_q_values, 1, next_best_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.policy_net.gamma * max_next_q_values * (~dones).float()
        
        loss = nn.functional.huber_loss(current_q_values, targets)

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

    def select_action(self, state: torch.tensor) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.policy_net.out_feats)
        else:
            return torch.argmax(self.policy_net.forward(state.unsqueeze(0))).item()
        
    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.decay_rate)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_agent(self, policy_path: str, target_path: str):
        self.policy_net.save_model(policy_path)
        self.target_net.save_model(target_path)

    def load_agent(policy_path: str, target_path: str, device: str):
        return Agent(
            DQN.load_model(policy_path, device),
            DQN.load_model(target_path, device),
            device
        )

def train(agent: Agent, device: str, env, num_episodes: int, policy_save_path: str, target_save_path: str, verbose: bool = True) -> list[float]:
    best_reward = float('-inf')
    timesteps_completed = 0
    rewards = []

    for i in range(num_episodes):
        state, _ = env.reset()
        state = state.to(device)
        done, truncated = False, False
        ep_reward = 0
        ep_timesteps = 0

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.memory.store((state, action, reward, next_state, done))
            state = next_state.to(device)

            if len(agent.memory.buffer) >= agent.batch_size:
                agent.update_policy()

            if timesteps_completed % 1000 == 0:
                agent.update_target()        

            ep_reward += reward
            timesteps_completed += 1
            ep_timesteps += 1

            if done or truncated:
                break

        agent.decay_epsilon()
        rewards.append(ep_reward)
        
        if ep_reward > best_reward:
            print(f'new best model with reward {ep_reward:.2f}, saving...')
            best_reward = ep_reward
            agent.save_agent(policy_save_path, target_save_path)

        if verbose:
            print(f'finished episode {i} with reward: {ep_reward:.2f}')
            print(f'episode timesteps: {ep_timesteps}')
            print(f'total timesteps completed: {timesteps_completed}')
            print('-' * 20)

    return rewards

def evaluate(agent: Agent, device, eval_env, eval_episodes):
    agent.epsilon = 0
    eval_rewards = 0

    for _ in range(eval_episodes):
        done, truncated = False, False
        state, _ = eval_env.reset()
        total_reward = 0

        while not (done or truncated):
            with torch.no_grad():
                state = state.to(device)
                action = agent.select_action(state)
                state, reward, done, truncated, _ = eval_env.step(action)
                total_reward += reward

        print("evaluation reward:", total_reward)
        eval_rewards += total_reward

    eval_env.close()
    print('average reward:', eval_rewards / eval_episodes)


