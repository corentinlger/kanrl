import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.optim import Adam

import gymnasium as gym
from gymnasium import spaces 


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000):

    env = gym.make(env_name)
    assert isinstance(env.observation_space, spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, spaces.Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)
    
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        observation, _ = env.reset()       
        done, truncated = False, False           
        ep_rews = []           

        # collect experience by acting in the environment with current policy
        while True:
            batch_obs.append(observation.copy())
            act = get_action(torch.as_tensor(observation, dtype=torch.float32))
            observation, reward, done, truncated, _ = env.step(act)
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(reward)
            if done or truncated:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                observation, _= env.reset()
                done, truncated, ep_rews = False, False, []
                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, lr=args.lr)
