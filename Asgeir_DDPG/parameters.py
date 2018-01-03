# Agent parameters
DEFAULT_EPISODES = 300
DEFAULT_STEPS = 1000
DEFAULT_ENVIRONMENT = 'Pendulum-v0'
DEFAULT_ACTION_FRAMES = 1
DEFAULT_LOAD = False

# DDPG learning parameters
DEFAULT_MEMORY_CAPACITY = 100000
DEFAULT_GAMMA = 0.99
DEFAULT_MINI_BATCH_SIZE = 64
DEFAULT_TAU = 0.001

# Neural network parameters
DEFAULT_LEARNING_RATE = 0.0001

# Actor network parameters
DEFAULT_ACTOR_LEARNING_RATE = 0.0001
DEFAULT_ACTOR_HIDDEN_LAYERS = [200, 100]

# Critic network parameters
DEFAULT_CRITIC_LEARNING_RATE = 0.001
DEFAULT_CRITIC_HIDDEN_LAYERS = [200, 100]


DEFAULT_ID = 0

def parse_args():
  agent_params = {'episodes': DEFAULT_EPISODES, 'steps': DEFAULT_STEPS, 'environment': DEFAULT_ENVIRONMENT, 'action_frames' : DEFAULT_ACTION_FRAMES, 'load' : DEFAULT_LOAD}
  ddpg_params = {'memory_capacity': DEFAULT_MEMORY_CAPACITY, 'gamma': DEFAULT_GAMMA, 'mini_batch_size': DEFAULT_MINI_BATCH_SIZE, 'tau' : DEFAULT_TAU}
  cnn_params = {'lr': DEFAULT_LEARNING_RATE, 'mini_batch_size': DEFAULT_MINI_BATCH_SIZE}
  
  a_params = {'a_lr': DEFAULT_ACTOR_LEARNING_RATE, 'a_hl': DEFAULT_ACTOR_HIDDEN_LAYERS}
  c_params = {'c_lr':DEFAULT_CRITIC_LEARNING_RATE, 'c_hl': DEFAULT_CRITIC_HIDDEN_LAYERS}
  
  assert len(a_params['a_hl']) > 1 and len(c_params['c_hl']) > 1

  return agent_params, ddpg_params, cnn_params, a_params, c_params