import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from torch.distributions.categorical import Categorical
from torch.optim import Adam
from gymnasium import spaces 
from kan import KAN

GRID = 5
WIDTH = 8

def kan_train(
    net,
    target,
    data,
    optimizer,
    gamma=0.99,
    lamb=0.0,
    lamb_l1=1.0,
    lamb_entropy=2.0,
    lamb_coef=0.0,
    lamb_coefdiff=0.0,
    small_mag_threshold=1e-16,
    small_reg_factor=1.0,
):
    def reg(acts_scale):
        def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = 0.0
        for i in range(len(acts_scale)):
            vec = acts_scale[i].reshape(
                -1,
            )

            p = vec / torch.sum(vec)
            l1 = torch.sum(nonlinear(vec))
            entropy = -torch.sum(p * torch.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(net.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(net.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(
                torch.mean(torch.abs(torch.diff(net.act_fun[i].coef)), dim=1)
            )
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    observations, actions, next_observations, rewards, terminations = data

    with torch.no_grad():
        next_q_values = net(next_observations)
        next_actions = next_q_values.argmax(dim=1)
        next_q_values_target = target(next_observations)
        target_max = next_q_values_target[range(len(next_q_values)), next_actions]
        td_target = rewards.flatten() + gamma * target_max * (
            1 - terminations.flatten()
        )

    old_val = net(observations).gather(1, actions).squeeze()
    loss = nn.functional.mse_loss(td_target, old_val)
    reg_ = reg(net.acts_scale)
    loss = loss + lamb * reg_
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000):

    env = gym.make(env_name)
    assert isinstance(env.observation_space, spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, spaces.Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Check how to return only logits
    kan_logits_net = KAN(
            width=[env.observation_space.shape[0], WIDTH, env.action_space.n],
            grid=GRID,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
        )

    def get_policy(obs):
        logits = (
                    kan_logits_net(obs.unsqueeze(0).double())
                    .squeeze()
                )
        return Categorical(logits=logits)
    
    def get_batch_policy(obs):
        logits = kan_logits_net(obs.double())
        return Categorical(logits=logits)
    
    def get_action(obs):
        action = get_policy(obs).sample().item()
        return action
    
    def compute_loss(obs, act, weights):
        logits = get_batch_policy(obs)
        logp = logits.log_prob(act) 
        return -(logp * weights).mean()
    
    optimizer = Adam(kan_logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []          
        batch_acts = []        
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

