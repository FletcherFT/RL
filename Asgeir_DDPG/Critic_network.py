import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn import relu, tanh

class Critic:
  """
  Critic Neural Network model.
  """

  def __init__(self, action_shape, observation_shape, scope, session, params):
    with tf.variable_scope(scope):
      self.scope = scope
      self.num_actions = action_shape[0]
      self.num_observations = observation_shape[0]
      self.lr = params['c_lr']
      self.hidden_layers = params['c_hl']
      self.session = session
      self.create_model()

  def add_placeholders(self):
    input_pl = tf.placeholder(tf.float32, shape=(None, self.num_observations))
    actions_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions))
    labels_pl = tf.placeholder(tf.float32, shape=(None, 1))
    return input_pl, actions_pl, labels_pl
 
  def nn(self, obs, act):
    for i in range(len(self.hidden_layers)):
      if i == 0:
        net = tf.layers.dense(inputs = obs, units = self.hidden_layers[i], activation = relu, name = 'CriticLayer' + str(1+i))
      elif i == 1:
        net = tf.concat([net, act], 1) # input actions at second hidden layer
        net = tf.layers.dense(inputs = net, units = self.hidden_layers[i], activation = relu, name = 'CriticLayer' + str(1+i))
      else:
        net = tf.layers.dense(inputs = net, units = self.hidden_layers[i], activation = relu, name = 'CriticLayer' + str(1+i))

    out = tf.layers.dense(inputs = net, units = 1, activation = None, name = 'CriticLayerOut')
    self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
    return out
      
  def create_model(self):
    self.input_pl, self.actions_pl, self.labels_pl = self.add_placeholders()
    self.q_vals = self.nn(self.input_pl, self.actions_pl)
    self.network_params = tf.trainable_variables()
    self.loss = tf.reduce_mean(tf.square(self.labels_pl - self.q_vals))
    self.grads = tf.gradients(self.loss, self.trainables)
    optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
    self.train_op = optimizer.minimize(self.loss)
    self.action_gradient = tf.gradients(self.q_vals, self.actions_pl)

  def assign_trainables(self, ts, tau=1.0):
    ops = []
    for i, t in enumerate(self.trainables):
      ops.append(t.assign((1-tau) * t.value() + tau * ts[i].value()))
    return ops
  
  def train_step(self, observations, labels, actions):
    loss, _, q_values = self.session.run(
      [self.loss, self.train_op, self.q_vals],
      feed_dict = {self.input_pl: observations, self.actions_pl: actions, self.labels_pl: labels})
    return loss

  def predict(self, observation, action):
    q_values = self.session.run(
      self.q_vals,
      feed_dict = {self.input_pl: observation, self.actions_pl: action})
    return q_values
  
  def get_action_gradient(self, observation, action):
    action_gradient = self.session.run(
      self.action_gradient,
      feed_dict = {self.input_pl: observation, self.actions_pl: action})
    return action_gradient
  
  def get_gradients(self, observation, labels, action):
    grads = self.session.run(
      self.grads,
      feed_dict = {self.input_pl: observation, self.actions_pl: action, self.labels_pl: labels})
    return grads
  
  def get_loss(self, observations, labels, actions):
    loss = self.session.run(
      self.loss,
      feed_dict = {self.input_pl: observations, self.actions_pl: actions, self.labels_pl: labels})
    return loss
