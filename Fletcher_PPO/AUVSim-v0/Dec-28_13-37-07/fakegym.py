import rospy
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
import tf.transformations as transform
import random
from math import pi
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
 
class env():
	def __init__(self,discrete=False,steplimit=500):
		self.state = None
		self.scale = 30.0
		self.nh = rospy.init_node('fakegyminterface')
		self.steps = 0
		self.steplimit = steplimit
		if discrete:
			self.action_space = spaces.Discrete(6)
		else:
			self.action_space = spaces.Box(low=-1, high=1, shape=(6,))
		self.observation_space = spaces.Box(low=-100, high=100, shape=(12,))
		self.spec = EnvSpec('AUVSim-v0',timestep_limit=self.steplimit)
		self.rng = random
		self.lastaction = [0.0,0.0,0.0,0.0,0.0,0.0]
		try:
			#Subscriber to get the model's pose & twist
			rospy.Subscriber('/brov/state', Odometry, self.updateState)
			#Publisher to set randomize the model pose & twist on each restart
			self.reset_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
			self.thrust_pub = rospy.Publisher('/brov/thruster_command', JointState, queue_size=1)
			self.step_srv = rospy.ServiceProxy('/gazebo/step', Empty)
			rospy.wait_for_service('/gazebo/step')
			self.step_srv()
		except rospy.ROSInterruptException:
			pass

	def updateState(self,newState):
		x = newState.pose.pose.position.x
		y = newState.pose.pose.position.y
		z = newState.pose.pose.position.z
		quat = (
			newState.pose.pose.orientation.x,
			newState.pose.pose.orientation.y,
			newState.pose.pose.orientation.z,
			newState.pose.pose.orientation.w,)
		euler = transform.euler_from_quaternion(quat)
		u = newState.twist.twist.linear.x
		v = newState.twist.twist.linear.y
		w = newState.twist.twist.linear.z
		p = newState.twist.twist.angular.x
		q = newState.twist.twist.angular.y
		r = newState.twist.twist.angular.z
		self.state = (x,y,z,euler[0],euler[1],euler[2],u,v,w,p,q,r,)

	def reset(self):
		#Message to send to the simulator
		zeroState = ModelState()
		zeroState.model_name = 'brov'
		zeroState.pose.position.x = self.rng.uniform(-100.0,100.0)
		zeroState.pose.position.y = self.rng.uniform(-100.0,100)
		zeroState.pose.position.z = self.rng.uniform(-200.0,0.0)
		r = self.rng.uniform(-pi,pi)
		p = self.rng.uniform(-pi,pi)
		y = self.rng.uniform(-pi,pi)
		quat = transform.quaternion_from_euler(r,p,y)
		zeroState.pose.orientation.x = quat[0]
		zeroState.pose.orientation.y = quat[1]
		zeroState.pose.orientation.z = quat[2]
		zeroState.pose.orientation.w = quat[3]
		zeroState.twist.linear.x = self.rng.uniform(-0.5,0.5)
		zeroState.twist.linear.y = self.rng.uniform(-0.5,0.5)
		zeroState.twist.linear.z = self.rng.uniform(-0.5,0.5)
		zeroState.twist.angular.x = self.rng.uniform(-0.1,0.1)
		zeroState.twist.angular.y = self.rng.uniform(-0.1,0.1)
		zeroState.twist.angular.z = self.rng.uniform(-0.1,0.1)
		#Send to the simulator
		self.reset_pub.publish(zeroState)
		#restart the step counter
		self.steps=0
		state = []
		state.append(zeroState.pose.position.x)
		state.append(zeroState.pose.position.y)
		state.append(zeroState.pose.position.z)
		state.append(r)
		state.append(p)
		state.append(y)
		state.append(zeroState.twist.linear.x )
		state.append(zeroState.twist.linear.y )
		state.append(zeroState.twist.linear.z )
		state.append(zeroState.twist.angular.x)
		state.append(zeroState.twist.angular.y)
		state.append(zeroState.twist.angular.z)
		self.state = state
		return np.array(state)

	def step(self,action):
		thruster_msg = JointState()
		thruster_msg.name = ['thr1','thr2','thr3','thr4','thr5','thr6']
		scaledaction = [self.scale*i for i in action]
		thruster_msg.position = [scaledaction[0],scaledaction[1],scaledaction[2],scaledaction[3],
		scaledaction[4],scaledaction[5]]
		self.lastaction = [scaledaction[0],scaledaction[1],scaledaction[2],scaledaction[3],
		scaledaction[4],scaledaction[5]]
		self.thrust_pub.publish(thruster_msg)
		rospy.wait_for_service('/gazebo/step')
		self.step_srv()
		self.steps+=1
		done = self.steps>self.steplimit# or (abs(self.state[3]-0.0)<1e-2 and abs(self.state[9])<1e-2)
		return np.array(self.state), self.reward(), done, "don't look here"

	def reward(self):
		rollerr = abs(self.state[3]-0.0)
		pitcherr = abs(self.state[4]-0.0)
		#pitcherr = 0
		yawerr = abs(self.state[5]-0.0)
		yawerr = 0
		RotationObj = 1.0/(rollerr+pitcherr+yawerr+1e-2)
		#EnergyObj = 1.0/(np.array(self.lastaction).sum()+1e-2)
		EnergyObj = 0.0
		return RotationObj+EnergyObj

	def seed(self,i):
		self.rng.seed(i)

	def close(self):
		pass
