import numpy as np
import random as random
import math
import tensorflow as tf
import pickle
import time
from collections import deque

from Actor_network import Actor
from Critic_network import Critic
import Noise as N

class DDPG:
  def __init__(self, action_shape, action_bound, observation_shape, ddpg_params, cnn_params, a_params, c_params, session):
    self.session = session
    self.gamma = ddpg_params['gamma']
    self.mini_batch_size = ddpg_params['mini_batch_size']
    self.tau = ddpg_params['tau']
    self.memory = deque(maxlen=ddpg_params['memory_capacity'])
    self.action_shape = action_shape
    self.actor_noise = N.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_shape[0]))
    
    # initialize networks
    self.A = Actor(action_shape, action_bound, observation_shape, 'Actor_main', self.session, self.mini_batch_size, a_params)
    self.A_t = Actor(action_shape, action_bound, observation_shape, 'Actor_target', self.session, self.mini_batch_size, a_params)
    self.A_copy_target = self.A_t.assign_trainables(self.A.trainables, tau = 1.0)
    self.A_upda_target = self.A_t.assign_trainables(self.A.trainables, self.tau)
        
    self.C = Critic(action_shape, observation_shape, 'Critic_main', self.session, c_params)
    self.C_t = Critic(action_shape, observation_shape, 'Critic_target', self.session, c_params)
    self.C_copy_target = self.C_t.assign_trainables(self.C.trainables, tau = 1.0)
    self.C_upda_target = self.C_t.assign_trainables(self.C.trainables, self.tau)
    
    print ("Networks initialized")

  def select_action(self, obs, stochastic = False, target = False, step = 0):
    """
    Selects the next action to take based on the current state and learned policy.
    Args:
      observation: the current state
    """
    if target: action = self.A_t.predict(obs)
    else: action = self.A.predict(obs)
    if stochastic: 
      nn = self.actor_noise()
      action += nn
    return action
  
  def update_state(self, observation, action, new_observation, reward, done):
    transition = {'observation': observation,
                  'action': action,
                  'new_observation': new_observation,
                  'reward': reward,
                  'is_done': done}
    self.memory.append(transition)

  def get_random_mini_batch(self):
    """
    Gets a random, unique sample of transitions from the replay memory.
    """
    rand_idxs = random.sample(range(len(self.memory)), self.mini_batch_size)
    mini_batch = []
    for idx in rand_idxs:
      mini_batch.append(self.memory[idx])

    return mini_batch
  
  def train_step(self):
    """
    Updates the actor and critic networks based on the mini batch
    """
    C_loss, C_difference, A_difference = 0, 0, 0
    if len(self.memory) > self.mini_batch_size:
      mini_batch = self.get_random_mini_batch()

      # Calculations for critic network
      observations, new_observations, actions, C_labels, is_done = [], [], [], [], []
      for sample in mini_batch:
        observations.append(sample['observation'])
        new_observations.append(sample['new_observation'])
        actions.append(sample['action'])
        C_labels.append(sample['reward'])
        is_done.append(sample['is_done'])

      new_actions = self.select_action(new_observations, stochastic = False, target = True)
      c_new_values = self.C_t.predict(new_observations, new_actions)
      for i in range(len(c_new_values)):
        # Latter is necissary to convert to array with dtybe of float 32
        C_labels[i] = C_labels[i] + self.gamma * c_new_values[i] if not is_done[i] else  C_labels[i] + 0 * c_new_values[i]
        
      # Convert appended calculations into an array
      observations = np.array(observations)
      actions = np.array(actions)
      C_labels = np.array(C_labels)
      
      # Train critic network
      C_loss = self.C.train_step(observations, C_labels, actions)     

      # Calculations for actor network
      acts = self.select_action(observations, stochastic = False, target = True)
      action_gradients = self.C.get_action_gradient(observations, acts)[0]
      grad = self.A.get_actor_gradient(observations, np.array(action_gradients))

      action_gradients = np.array(action_gradients)
      grad = np.array(grad)
      
      # Train actor network
      self.A.train_step(observations, action_gradients)
      
      # Slow update the target networks
      self.session.run(self.C_upda_target)
      self.session.run(self.A_upda_target)
      
      # Compute difference between main and target networks      
      C_pred = self.C.predict(observations, actions)
      C_t_pred = self.C_t.predict(observations, actions)
      C_difference = np.mean((C_pred - C_t_pred)**2)
      
      A_pred = self.select_action(new_observations, stochastic = False, target = False)
      A_t_pred = self.select_action(new_observations, stochastic = False, target = True)
      A_difference = np.mean((A_pred - A_t_pred)**2)

    return C_loss, C_difference, A_difference
    
  def save(self, saver, env):
    try:
      saver.save(self.session, "Trainings/" + env + "/" + time.strftime("%d%m%Y") + "/model.ckpt")
      with open("Trainings/" + env + "/" + time.strftime("%d%m%Y") + "/memory.txt", "wb") as fp: pickle.dump(self.memory, fp) # Pickling
      print ("Networks saved")
    except Exception:
      print ("Could not save!")

  def load(self, saver, env, date = None):
    if date == None:
      saver.restore(self.session, "Trainings/" + env + "/" + time.strftime("%d%m%Y") + "/model.ckpt")
      with open("Trainings/" + env + "/" + time.strftime("%d%m%Y") + "/memory.txt", "rb") as fp: self.memory = pickle.load(fp) # Unpickling
    else:
      saver.restore(self.session, "Trainings/" + env + "/" + date + "/model.ckpt")
      with open("Trainings/" + env + "/" + date + "/memory.txt", "rb") as fp: self.memory = pickle.load(fp) # Unpickling
    print ("Networks loaded")
