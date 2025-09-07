import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch as T
import torch.optim as opt
import sys

class Memory:
    def __init__(self, batch_size: int):
        self.states = []
        self.state_vals = []
        self.next_states = []
        self.next_state_vals = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def add(self, state: T.tensor, 
            state_val: T.tensor, 
            next_state: T.tensor,
            next_state_val: T.tensor,
            action: T.tensor,
            prob: T.tensor,
            reward: T.tensor,
            done: T.tensor,
            ) -> None:
        self.states.append(state)
        self.state_vals.append(state_val.squeeze(-1))
        self.next_states.append(next_state)
        self.next_state_vals.append(next_state_val.squeeze(-1))
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_batch(self) -> tuple:
        start = T.arange(0, len(self.states), 32)
        indices = T.randperm(len(self.states))
        batches = [indices[i:i + self.batch_size] for i in start]
        return T.stack(self.states), T.stack(self.state_vals), T.stack(self.next_states), T.stack(self.next_state_vals), T.stack(self.actions), T.stack(self.probs), T.stack(self.rewards), batches

    def clear_memory(self) -> None:
        self.states.clear()
        self.state_vals.clear()

        self.next_states.clear()
        self.next_state_vals.clear()

        self.actions.clear()
        self.probs.clear()
        self.rewards.clear()
        self.dones.clear()


class Actor(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super(Actor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, states: T.tensor) -> T.tensor:
        logits = self.layers(states)
        return Categorical(logits = logits)


class Critic(nn.Module):
    def __init__(self,  in_features: int, out_features: int, hidden_size: int):
        super(Critic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        )
    
    def forward(self, states: T.tensor) -> T.tensor:
        return self.layers(states)


class Agent():
    def __init__(self, actor: Actor, critic: Critic, epsilon: float, gamma: float, lam: float, actor_lr: float, critic_lr: float, device: str, batch_size: int):
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_opt = opt.Adam(actor.parameters())
        self.critic_opt = opt.Adam(critic.parameters())
        self.device = device
        self.memory = Memory(batch_size)
        self.all_rewards = []
        self.all_steps = []

    def save_agent(self, path: str) -> None:
        T.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'hyperparameters': {
                'actor_lr': self.actor_lr,
                'critic_lr': self.critic_lr,
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'lam': self.lam,
                'actor_in_feats': self.actor.in_features,
                'actor_out_feats': self.actor.out_features,
                'critic_in_feats': self.critic.in_features,
                'critic_out_feats': self.critic.out_features
            }
        }, path)

    @staticmethod
    def load_agent(env, path: str) -> 'Agent':
        chckpt = T.load(path)

        actor = Actor(chckpt['hyperparameters']['actor_in_feats'], chckpt['hyperparameters']['actor_out_feats'])
        critic = Critic(chckpt['hyperparameters']['critic_in_feats'], chckpt['hyperparameters']['critic_out_feats'])
        actor.load_state_dict(chckpt['actor'])
        critic.load_state_dict(chckpt['critic'])

        return Agent(
            env, 
            actor, 
            critic, 
            chckpt['hyperparameters']['epsilon'],
            chckpt['hyperparameters']['gamma'],
            chckpt['hyperparameters']['lam'],
            chckpt['hyperparameters']['actor_lr'],
            chckpt['hyperparameters']['critic_lr']
        )

    def select_action(self, states: T.tensor) -> tuple:
        dist = self.actor.forward(states)
        actions = dist.sample()

        return actions, dist.log_prob(actions)  
    
    def get_state_values(self, states: T.tensor) -> T.tensor:
        return self.critic.forward(states)

    def update_nets(self, actor_loss: T.tensor, critic_loss: T.tensor) -> None:
        total_loss = actor_loss + 0.5 * critic_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()

    def fit(self, K: int) -> None:        
        states, state_vals, next_states, next_state_vals, actions, probs, rewards, batches = self.memory.get_batch()
        length = len(states)

        advantage = 0
        advantages = []

        for t in reversed(range(length)):
            delta = rewards[t].to(self.device) + self.gamma * next_state_vals[t] - state_vals[t]
            advantage = delta + self.gamma * self.lam * advantage
            advantages.append(advantage)

        advantages = advantages[::-1]
        advantages = T.stack(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(K):
            for batch in batches:
                b_states = states[batch].detach()
                b_state_vals = state_vals[batch].detach()
                b_old_probs = probs[batch].detach() # get action probabilites from samples
                b_actions = actions[batch] # get selected actions from samples
                b_advantages = advantages[batch].detach()

                dists = self.actor.forward(b_states)
                new_probs = dists.log_prob(b_actions)
                ratio = (new_probs - b_old_probs).exp()

                G_vals = b_advantages + b_state_vals

                critic_loss = ((G_vals.detach() - self.get_state_values(b_states)) ** 2).mean() # loss for critic network
                actor_loss = -T.min(ratio * b_advantages, 
                                    T.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * b_advantages).mean()
                
                self.update_nets(actor_loss, critic_loss)


