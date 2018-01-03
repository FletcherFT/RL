"""
Main Script for Running PPO Algorithm on AUVSim-v0

Adapted from work by Patrick Coady (pat-coady.github.io)
Author:  Fletcher Thompson
"""

import fakegym
import numpy as np
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import tensorflow as tf
from IPython.lib import backgroundjobs as bg

## FUNCTION DECLARATIONS ##
def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew

def add_value(sess, trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(sess,observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages

def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return observes, actions, advantages, disc_sum_rew

def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })

def run_episode(sess,env, policy, scaler):
    """ Run single episode

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(sess, obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 5e-2  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(sess, env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    print("Running Policy for {} Episodes".format(episodes))
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(sess, env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        print("Episode: {}".format(e))
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})
    return trajectories

## RUN THE ENVIRONMENT ##
env = fakegym.env()
jobs = bg.BackgroundJobManager()
# run the interface as a background process
jobs.new('env.run()')

## SET HYPERPARAMETERS ##
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
env_name="AUVSim-v0"
model_dir = "tmp/model/"
hid1_mult=10 # sets size of first hidden layer as a multiple of observation dimension size
kl_targ=0.003 # sets the maximum allowable KL divergence for the policy update
policy_logvar=-1.0 # sets the initial log variance of the policy
num_episodes = 50000 # stopping condition (total number of episodes)
batch_size = 20 # number of trajectories to generate for each training session
gamma = 0.995 # discount factor for future reward summation
lam = 0.98 # general advantage estimation

obs_dim += 1  # add 1 to the obs dimension for time
now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
logger = Logger(logname=env_name, now=now)
scaler = Scaler(obs_dim)

## BUILD THE NETWORKS ##
tf.reset_default_graph()
G = tf.Graph()
val_func = NNValueFunction(G,obs_dim, hid1_mult)
policy = Policy(G, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)

## INITIALIZE THE NETWORK VARIABLES ##
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
with G.as_default():
    sess = tf.Session(graph=G,config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
writer = tf.summary.FileWriter("./tmp/log", G) # to visualise the graph through tensorboard
print("***\nINFO:  tensorboard visualisation\nrun 'tensorboard --logdir=tmp' in another terminal (from this directory) to visualise the networks\n***")
# run a few episodes of random policy for observation normalisation:
print("Generating random states to initialise the observation scaler")
run_policy(sess,env, policy, scaler, logger, episodes=5)
print("Beginning Training")
episode = 0
while episode < num_episodes:
    print("Episode {} out of {}".format(episode,num_episodes))
    trajectories = run_policy(sess,env, policy, scaler, logger, episodes=batch_size)
    episode += len(trajectories)
    add_value(sess, trajectories, val_func)  # add estimated values to episodes
    add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of rewards
    add_gae(trajectories, gamma, lam)  # calculate advantage
    # concatenate all episodes into single NumPy arrays
    observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
    # add various stats to training log:
    log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
    policy.update(sess,observes, actions, advantages, logger)  # update policy
    val_func.fit(sess,observes, disc_sum_rew, logger)  # update value function
    logger.write(display=True)  # write logger results to file and stdout
    if episode % 1000 == 0:
        saver.save(sess, model_dir+env_name, global_step=episode) #save the model every 1000 episodes
        print("Saved Network")
logger.close()
sess.close()
env.close()