import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch as T
import torch.optim as opt


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
        start = T.arange(0, len(self.states), self.batch_size)
        indices = T.randperm(len(self.states))
        batches = [indices[i:i + self.batch_size] for i in start]
        return T.stack(self.states), T.stack(self.state_vals), T.stack(self.next_states), T.stack(self.next_state_vals), T.stack(self.actions), T.stack(self.probs), T.stack(self.rewards), T.stack(self.dones), batches

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
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
        self.hidden_size = hidden_size
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
    def __init__(self, actor: Actor, critic: Critic, epsilon: float, gamma: float, lam: float, c1: float, c2: float, actor_lr: float, critic_lr: float, device: str, batch_size: int, tuning: bool = False):
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_opt = opt.AdamW(actor.parameters(), lr = actor_lr)
        self.critic_opt = opt.AdamW(critic.parameters(), lr = critic_lr)
        self.device = device
        self.memory = Memory(batch_size)
        self.c1 = c1
        self.c2 = c2
        self.all_rewards = []
        self.all_steps = []
        self.tuning = tuning

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
                'actor_hs': self.actor.hidden_size,
                'critic_in_feats': self.critic.in_features,
                'critic_out_feats': self.critic.out_features,
                'critic_hs': self.critic.hidden_size,
                'batch_size': self.memory.batch_size,
                'c1': self.c1,
                'c2': self.c2,
            }
        }, path)

    @staticmethod
    def load_agent(path: str, device: str) -> 'Agent':
        chckpt = T.load(path, map_location = T.device(device))

        actor = Actor(chckpt['hyperparameters']['actor_in_feats'], chckpt['hyperparameters']['actor_out_feats'], chckpt['hyperparameters']['actor_hs']).to(device)
        critic = Critic(chckpt['hyperparameters']['critic_in_feats'], chckpt['hyperparameters']['critic_out_feats'], chckpt['hyperparameters']['critic_hs']).to(device)
        actor.load_state_dict(chckpt['actor'])
        critic.load_state_dict(chckpt['critic'])

        return Agent(
            actor, 
            critic, 
            chckpt['hyperparameters']['epsilon'],
            chckpt['hyperparameters']['gamma'],
            chckpt['hyperparameters']['lam'],
            chckpt['hyperparameters']['c1'],
            chckpt['hyperparameters']['c2'],
            chckpt['hyperparameters']['actor_lr'],
            chckpt['hyperparameters']['critic_lr'],
            device,
            chckpt['hyperparameters']['batch_size']
        )

    def select_action(self, states: T.tensor) -> tuple:
        dist = self.actor.forward(states)
        actions = dist.sample()

        return actions, dist.log_prob(actions)  
    
    def get_state_values(self, states: T.tensor) -> T.tensor:
        return self.critic.forward(states)

    def update_nets(self, actor_loss: T.tensor, critic_loss: T.tensor, entropy: T.tensor) -> None:
        actor_loss = actor_loss - self.c2 * entropy
        self.actor_opt.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        critic_loss = self.c1 * critic_loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

    def fit(self, K: int, num_envs: int) -> None:        
        states, state_vals, _, next_state_vals, actions, probs, rewards, dones, batches = self.memory.get_batch()
        length = len(states)

        num_steps = length // num_envs

        state_vals = state_vals.detach()
        next_state_vals = next_state_vals.detach()

        state_vals = state_vals.view(num_steps, num_envs)
        state_vals = state_vals.to(self.device)
        next_state_vals = next_state_vals.view(num_steps, num_envs)
        next_state_vals = next_state_vals.to(self.device)
        rewards = rewards.view(num_steps, num_envs)
        rewards = rewards.to(self.device)
        dones = dones.view(num_steps, num_envs)
        dones = dones.to(self.device)

        # compute gae per environment
        advantages = T.zeros_like(rewards).to(self.device)
        gae = T.zeros(num_envs).to(self.device)

        for t in reversed(range(num_steps)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * mask * next_state_vals[t] - state_vals[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        advantages = advantages.view(-1)
        state_vals = state_vals.view(-1)
        returns = advantages + state_vals

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(K):
            for batch in batches:
                b_states = states[batch]
                # b_state_vals = state_vals[batch].detach()
                b_old_probs = probs[batch].detach() # get action probabilites from samples
                b_actions = actions[batch] # get selected actions from samples
                b_advantages = advantages[batch].detach()

                dists = self.actor.forward(b_states)
                new_probs = dists.log_prob(b_actions)
                ratio = (new_probs - b_old_probs).exp()

                critic_loss = ((returns[batch] - self.get_state_values(b_states).squeeze()) ** 2).mean()
                actor_loss = -T.min(ratio * b_advantages, 
                                T.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * b_advantages).mean()
                
                self.update_nets(actor_loss, critic_loss, dists.entropy().mean())


def train(env, agent: Agent, num_envs: int, train_iters: int, timesteps: int, K: int, save_path: str, verbose: bool = False):
    best_reward = float('-inf')
    
    for train_iter in range(train_iters):
        ep_reward = 0
        ep_steps = 0

        states, _ = env.reset()
        
        for _ in range(timesteps):
            states = states.to(agent.device)
            actions, probs = agent.select_action(states)
            state_vals = agent.get_state_values(states)

            next_states, rewards, dones, terminated, _ = env.step(actions.cpu())
            next_states = next_states.to(agent.device)

            ep_reward += sum(rewards).item()

            next_state_vals = agent.get_state_values(next_states)
            for state, state_val, next_state, next_state_val, action, prob, reward, done in zip(states, state_vals, next_states, next_state_vals, actions, probs, rewards, dones):
                agent.memory.add(state, state_val, next_state, next_state_val, action, prob, reward, done)

            states = next_states
            ep_steps += 1
        
        if verbose:
            print('finished episode:', train_iter)
            print('total reward:', ep_reward)
            print('number of steps:', ep_steps)
            print('-' * 15)

        agent.fit(K, num_envs)
        agent.memory.clear_memory()
        agent.all_rewards.append(ep_reward)
        agent.all_steps.append(ep_steps)

        if ep_reward > best_reward and not agent.tuning:
            best_reward = ep_reward
            agent.save_agent(save_path)
            print('new best model... saving...')
    
    if verbose:
        print('finished training with best reward:', best_reward)


def evaluate(eval_env, eval_episodes, agent):
    eval_rewards = 0

    for _ in range(eval_episodes):
        done, truncated = False, False
        state, _ = eval_env.reset()
        total_reward = 0

        while not (done or truncated):
            with T.no_grad():
                state = state.to(agent.device)
                dist = agent.actor(state)
                action = dist.probs.argmax()
                state, reward, done, truncated, _ = eval_env.step(action.unsqueeze(0).cpu())
                total_reward += reward

        print("evaluation reward:", total_reward.item())
        eval_rewards += total_reward.item()

    eval_env.close()
    print('average reward:', eval_rewards / eval_episodes)

