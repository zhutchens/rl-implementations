from src.ppo import Agent, Actor, Critic

def return_obj(env, device, train_iters: int, timesteps: int, K: int, train_func: callable, params: dict[str, tuple]):
    def objective(trial):
        trials = {}
        for param, info in params.items():
            if info[2] == 'int':
                trials[param] = trial.suggest_int(param, info[0], info[1])
            elif info[2] == 'float':
                trials[param] = trial.suggest_float(param, info[0], info[1])

        agent = Agent(
            Actor(trials['actor_in_feats'], trials['actor_out_feats'], trials['actor_hs']).to(device), 
            Critic(trials['critic_in_feats'], trials['critic_out_feats'], trials['critic_hs']).to(device),
            trials['epsilon'],
            trials['gamma'],
            trials['lambda'],
            trials['c1'],
            trials['c2'],
            trials['actor_lr'],
            trials['critic_lr'],
            device,
            trials['batch_size'],
            True
        )
        train_func(env, agent, train_iters, timesteps, K, False)

        return sum(agent.all_rewards)
    return objective


