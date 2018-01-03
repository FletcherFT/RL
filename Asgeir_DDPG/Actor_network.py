import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn import relu, tanh

class Actor:
  """
  Actor Neural Network model.
  """

  def __init__(self, action_shape, action_bound, observation_shape, scope, session, batch_size, params):
    with tf.variable_scope(scope):
      self.scope = scope
      self.num_actions = action_shape[0]
      self.action_bound = action_bound
      self.num_observations = observation_shape[0]
      self.lr = params['a_lr']
      self.hidden_layers = params['a_hl']
      self.batch_size = batch_size
      self.session = session
      self.create_model()
      
  def add_placeholders(self):
    input_pl = tf.placeholder(tf.float32, shape=(None, self.num_observations))
    action_gradient_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions))
    return input_pl, action_gradient_pl
    
  def nn(self, obs):
    for i in range(len(self.hidden_layers)):
      if i == 0:
        net = tf.layers.dense(inputs = obs, units = self.hidden_layers[i], activation = relu, name = 'ActorLayer' + str(1+i))
      else:
        net = tf.layers.dense(inputs = net, units = self.hidden_layers[i], activation = relu, name = 'ActorLayer' + str(1+i))
    out = tf.layers.dense(inputs=net, units=self.num_actions, activation=tanh, name='ActorOutputLayer')
    scaled_out = tf.multiply(out, self.action_bound)
    self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
    return out, scaled_out

  def create_model(self):
    self.input_pl, self.action_gradient_pl = self.add_placeholders()
    out, scaled_out = self.nn(self.input_pl)
    self.scaled_action = scaled_out
    self.unnormalized_actor_gradients  = tf.gradients(self.scaled_action, self.trainables, -self.action_gradient_pl)
    self.actor_gradients = tf.tuple(list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients)))
    optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
    self.train_op = optimizer.apply_gradients(zip(self.actor_gradients, self.trainables))

  def assign_trainables(self, ts, tau=1.0):
    ops = []
    for i, t in enumerate(self.trainables):
      ops.append(t.assign((1-tau) * t.value() + tau * ts[i].value()))
    return ops
  
  def train_step(self, observations, a_gradient):
    _, scaled_out = self.session.run(
      [self.train_op, self.scaled_action],
      feed_dict = {self.input_pl: observations, self.action_gradient_pl: a_gradient})    
  
  def predict(self, observation):
    scaled_out = self.session.run(
      self.scaled_action,
      feed_dict = {self.input_pl: observation})

    return scaled_out
  
  def get_actor_gradient(self, observations, action_gradient):
    actor_gradients = self.session.run(
      self.actor_gradients,
      feed_dict = {self.input_pl: observations, self.action_gradient_pl: action_gradient})

    return actor_gradients
    
  def get_un_actor_gradient(self, observations, action_gradient):
    actor_gradients = self.session.run(
      self.unnormalized_actor_gradients,
      feed_dict = {self.input_pl: observations, self.action_gradient_pl: action_gradient})

    return actor_gradients