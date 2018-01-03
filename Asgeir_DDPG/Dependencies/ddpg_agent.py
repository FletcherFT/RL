import gym
from parameters import parse_args
from ddpg import DDPG
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time
import os

def run_ddpg():
  agent_params, ddpg_params, cnn_params, a_params, c_params = parse_args()

  env = gym.make(agent_params['environment'])
  episodes = agent_params['episodes']
  action_frames = agent_params['action_frames']
  steps = env.spec.timestep_limit if agent_params['steps'] > env.spec.timestep_limit else agent_params['steps']
  action_shape = env.action_space.shape
  action_bound = env.action_space.high
  observation_shape = env.observation_space.shape
  tf.reset_default_graph()
  
  # Initialize buffer that stores Critic network losses, training and validation rewards
  statistics = []
  if not os.path.exists("Trainings/" + agent_params['environment'] + "/" + time.strftime("%d%m%Y")):
      os.makedirs("Trainings/" + agent_params['environment'] + "/" + time.strftime("%d%m%Y"))
  
  with tf.Session() as session:
    ddpg = DDPG(action_shape, action_bound, observation_shape, ddpg_params, cnn_params, a_params, c_params, session)
    saver = tf.train.Saver()
    
    # Load the network and previously recorded statistics
    if agent_params['load'] == True:
      ddpg.load(saver, agent_params['environment'])
      with open("Trainings/" + agent_params['environment'] + "/" + time.strftime("%d%m%Y") + "/statistics.txt", "rb") as fp:   # Unpickling
        statistics = pickle.load(fp)
    # Initialize ddpg learning
    else:
      ddpg.session.run(tf.global_variables_initializer())
      ddpg.session.run(ddpg.C_copy_target)
      ddpg.session.run(ddpg.A_copy_target)
      
    print ("Prefilling memory")
    observation = env.reset()
    while len(ddpg.memory) < ddpg.memory.maxlen:
        action = env.action_space.sample()
        new_observation, reward, done, _ = env.step(action)
        ddpg.update_state(observation, action, new_observation, reward, done)
        observation = new_observation if not done else env.reset()
            
    for i_episode in range(len(statistics), episodes):
      # Select an initial observation from distribution of initial state
      env.seed(i_episode)
      observation = env.reset()
      frames = action_frames
      
      # Initialize buffers that stores the reward of a latest episode and the average loss of an episode
      r_training, C_loss_array, C_diff_array, A_diff_array = 0, [], [], []

      for i in range(steps):
#        if i_episode % 10 == 0: env.render()
        if i_episode % 10 == 0 and i == 0 and i_episode != 0: 
          ddpg.save(saver, agent_params['environment'])
          try:
            with open("Trainings/" + agent_params['environment'] + "/" + time.strftime("%d%m%Y") + "/statistics.txt", "wb") as fp:   #Pickling
              pickle.dump(statistics, fp)
          except Exception:
            print ("Could not save!")

        # Select action based on the model
        if frames == action_frames:
          action = ddpg.select_action([observation], stochastic = True, target = False, step = i)[0]
          frames = 0
        new_observation, reward, done, _ = env.step(action)
        
#        # Reward shaping
#        if agent_params['environment'] == 'CartPole-v0':
#          reward = reward - np.absolute(new_observation[0])/2.4
#          if agent_params['environment'] == 'MountainCar-v0':
#            if i == 0: max_distance = 1
#            max_distance = max(0, min(max_distance, np.absolute(new_observation[0] - 0.5)/1.7))
#            if done: reward = -max_distance
            
        # Update the state
        frames += 1
        if frames == action_frames:
          ddpg.update_state(observation, action, new_observation, reward, done)
          _C_loss, _C_diff, _A_diff = ddpg.train_step()
          C_loss_array.append(_C_loss)
          C_diff_array.append(_C_diff)
          A_diff_array.append(_A_diff)
          r_training += reward
  
        if done:   
          if frames != action_frames:
            ddpg.update_state(observation, action, new_observation, reward, done)
            _C_loss, _C_diff, _A_diff = ddpg.train_step()
            C_loss_array.append(_C_loss)
            C_diff_array.append(_C_diff)
            A_diff_array.append(_A_diff)
            r_training += reward
          frames = action_frames
          break
        
        # Observation becomes new observation
        observation = new_observation
      
      # Calculate the average loss of previous training episode
      C_loss = np.mean(C_loss_array)
      C_diff = np.mean(C_diff_array)
      A_diff = np.mean(A_diff_array
                       )
      # Validation
      r_validation = 0
      observation = env.reset()
      frames = action_frames
      
      for t in range(steps):
#        if i_episode % 10 == 0: env.render()
        
        if frames == action_frames:
          action = ddpg.select_action([observation], False, False)[0]
          frames = 0
        new_observation, reward, done, _ = env.step(action)
  
        # Reward shaping
#        if agent_params['environment'] == 'CartPole-v0':
#          reward = reward - np.absolute(new_observation[0])/2.4
#          if agent_params['environment'] == 'MountainCar-v0':
#            if i == 0: max_distance = 1
#            max_distance = max(0, min(max_distance, np.absolute(new_observation[0] - 0.5)/1.7))
#            if done: reward = -max_distance
          
        frames += 1
        if frames == action_frames: r_validation += reward
        observation = new_observation
        if done: 
          if frames != action_frames: r_validation += reward
          break    
        
      statistics.append((r_training, r_validation, C_loss, C_diff, A_diff))     
      print('%4d. training reward: %6.2f, validation reward: %6.2f, C loss: %7.4f, C diff: %7.4f, A diff: %7.4f' % (i_episode+1, r_training, r_validation, C_loss, C_diff, A_diff))

    # env.monitor.close()
    ddpg.save(saver, agent_params['environment'])
    try:
      with open("Trainings/" + agent_params['environment'] + "/" + time.strftime("%d%m%Y") + "/statistics.txt", "wb") as fp:   #Pickling
        pickle.dump(statistics, fp)
    except Exception:
      print ("Could not save!")

    # Plot training statistics
    statistics = np.array(statistics).T
    mean_training_rewards = statistics[0]
    mean_validation_rewards = statistics[1]
    C_losses = statistics[2]
    C_diffs = statistics[3]
    A_diffs = statistics[4]
    
    plt.subplot(221)
    plt.plot(C_losses, label='C loss')
    plt.xlabel('epoch'); plt.ylabel('loss')
    plt.xlim((0, len(C_losses)))
    plt.legend(loc=1); plt.grid()
    
    plt.subplot(222)
    plt.plot(C_diffs, label='C diff')
    plt.xlabel('epoch'); plt.ylabel('difference')
    plt.xlim((0, len(C_diffs)))
    plt.legend(loc=1); plt.grid()
    
    plt.subplot(223)
    plt.plot(A_diffs, label='A diff')
    plt.xlabel('epoch'); plt.ylabel('difference')
    plt.xlim((0, len(A_diffs)))
    plt.legend(loc=1); plt.grid()

    plt.subplot(224)
    plt.plot(mean_training_rewards, label='mean training reward')
    plt.plot(mean_validation_rewards, label='mean validation reward')
    plt.xlabel('epoch'); plt.ylabel('mean reward')
    plt.xlim((0, len(mean_validation_rewards)))
    plt.legend(loc=4); plt.grid()
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
  run_ddpg()
  
