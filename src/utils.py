import torch
from src.ppo import Agent

def train(env, agent: Agent, train_iters: int, timesteps: int, K: int):
    for train_iter in range(train_iters):
            ep_reward = 0
            ep_steps = 0

            states, _ = env.reset()
            states = torch.from_numpy(states).to(agent.device)

            for _ in range(timesteps):
                actions, probs = agent.select_action(states)
                state_vals = agent.get_state_values(states)

                next_states, rewards, dones, terminated, _ = env.step(actions.detach().cpu().numpy())
                next_states = torch.from_numpy(next_states).to(agent.device)
                rewards = torch.from_numpy(rewards)
                dones = torch.from_numpy(dones)

                ep_reward += sum(rewards).item()

                next_state_vals = agent.get_state_values(next_states)
                for state, state_val, next_state, next_state_val, action, prob, reward, done in zip(states, state_vals, next_states, next_state_vals, actions, probs, rewards, dones):
                     agent.memory.add(state, state_val, next_state, next_state_val, action, prob, reward, done)

                states = next_states
                ep_steps += 1
            
            print('finished episode:', train_iter)
            print('total reward:', ep_reward)
            print('number of steps:', ep_steps)
            print('-' * 15)

            agent.fit(K)

            agent.all_rewards.append(ep_reward)
            agent.all_steps.append(ep_steps)

