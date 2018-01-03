import rospy
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
import tf.transformations as transform
import random
from math import pi, sqrt
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
 
class env():
	def __init__(self,discrete=False,steplimit=1000):
		self.state = np.zeros(18)
		self.scale = 30.0
		self.nh = rospy.init_node('fakegyminterface')
		self.steps = 0
		self.steplimit = steplimit
		if discrete:
			self.action_space = spaces.Discrete(6)
		else:
			self.action_space = spaces.Box(low=-1, high=1, shape=(6,))
		self.observation_space = spaces.Box(low=-200, high=200, shape=(18,))
		self.spec = EnvSpec('AUVSim-v0',timestep_limit=self.steplimit)
		self.rng = random
		self.lastaction = [0.0,0.0,0.0,0.0,0.0,0.0]
		try:
			#Subscriber to get the model's pose & twist
			rospy.Subscriber('/brov/state', Odometry, self.updateState)
			#Publisher to set randomize the model pose & twist on each restart
			self.reset_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
			self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=1)
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
		self.state = np.array([x,y,z,euler[0],euler[1],euler[2],u,v,w,p,q,r]+self.state[12:].tolist())

	def reset(self):
		#AUV State Message to send to the simulator
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
		#AUV Target Message to send to the simulator
		target = PoseStamped()
		target.pose.position.x = self.rng.uniform(-5,5)+zeroState.pose.position.x
		target.pose.position.y = self.rng.uniform(-5,5)+zeroState.pose.position.y
		target.pose.position.z = self.rng.uniform(-5,5)+zeroState.pose.position.z
		rd = self.rng.uniform(-0.1,0.1)
		pd = self.rng.uniform(-0.1,0.1)
		yd = self.rng.uniform(-pi,pi)
		quat = transform.quaternion_from_euler(rd,pd,yd)
		target.pose.orientation.x = quat[0]
		target.pose.orientation.y = quat[1]
		target.pose.orientation.z = quat[2]
		target.pose.orientation.w = quat[3]
		target.header.stamp = rospy.Time.now()
		target.header.frame_id = 'world'
		self.target_pub.publish(target)
		#Return the reset state
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
		state.append(target.pose.position.x)
		state.append(target.pose.position.y)
		state.append(target.pose.position.z)
		state.append(rd)
		state.append(pd)
		state.append(yd)
		self.state = np.array(state)
		#restart the step counter
		self.steps=0
		return self.state

	def step(self,action):
		#update the thrusts of the AUV
		thruster_msg = JointState()
		thruster_msg.name = ['thr1','thr2','thr3','thr4','thr5','thr6']
		scaledaction = [self.scale*i for i in action]
		thruster_msg.position = [scaledaction[0],scaledaction[1],scaledaction[2],scaledaction[3],
		scaledaction[4],scaledaction[5]]
		self.lastaction = [scaledaction[0],scaledaction[1],scaledaction[2],scaledaction[3],
		scaledaction[4],scaledaction[5]]
		thruster_msg.header.stamp = rospy.Time.now()
		thruster_msg.header.frame_id = 'base_link'
		self.thrust_pub.publish(thruster_msg)
		#perform the stepping operation
		rospy.wait_for_service('/gazebo/step')
		self.step_srv()
		#increment the step by 1
		self.steps+=1
		#calculate the stopping criteria
		distance2target = sqrt((self.state[0]-self.state[12])**2
			+(self.state[1]-self.state[13])**2+(self.state[2]-self.state[14])**2)
		angle2target = sqrt((self.state[3]-self.state[15])**2+(self.state[4]-self.state[16])**2
			+(self.state[5]-self.state[17])**2)
		speed2target = sqrt((self.state[6])**2+(self.state[7])**2+(self.state[8])**2)
		angspeed2target =sqrt((self.state[9])**2+(self.state[10])**2+(self.state[11])**2)
		#done is true if the steps are greater than the limit, or if the distance, angles and speeds are within acceptable limits
		done = self.steps>self.steplimit or (distance2target<0.1 and angle2target<0.05 and speed2target<0.1 and angspeed2target<0.05)
		return np.array(self.state), self.reward(), done, "don't look here"

	def reward(self):
		rollerr = abs(self.state[3]-self.state[15])
		pitcherr = abs(self.state[4]-self.state[16])
		yawerr = abs(self.state[5]-self.state[17])
		xerr = abs(self.state[0]-self.state[12])
		yerr = abs(self.state[1]-self.state[13])
		zerr = abs(self.state[2]-self.state[14])
		# MAXIMUM REWARD PER STEP = 1 (eroll epitch eyaw ex ey ez = 0)
		PoseObj = 1.0/(rollerr+pitcherr+yawerr+xerr+yerr+zerr+1)
		TimeObj = -1.0
		#RotationObj = 1.0/(rollerr+pitcherr+yawerr+1e-2)
		#EnergyObj = 1.0/(np.array(self.lastaction).sum()+1e-2)
		# MAXIMUM REWARD PER STEP = 1 + -1 = 0 -> TOTAL REWARD FOR 1 EPISODE RANGES FROM -960 to 0
		return PoseObj+TimeObj#+RotationObj+EnergyObj

	def seed(self,i):
		self.rng.seed(i)

	def close(self):
		rospy.signal_shutdown("close method called, stopping node")