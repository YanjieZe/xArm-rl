import numpy as np
import os
import env.robot.reward_utils as reward_utils
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class ShelfPlacingEnv(BaseEnv, utils.EzPickle):
	"""
	Place the object on the shelf
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False, env_randomness_scale=.3):
		self.sample_large = 1
		self.env_randomness_scale = env_randomness_scale
		
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz,
			has_object=True
		)
		
		self.flipbit = 1
		self.state_dim = (17,)
		# 2022.01.19: 0.05 -> 0.1
		self.distance_threshold = 0.05
		
		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):
    		
		actions = self.current_action
		
		objPos = self.sim.data.get_site_xpos('object0').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		# heightTarget = self.lift_height + self.objHeight
		heightTarget = self.sim.data.get_site_xpos('target0')[-1] - 0.05 # 2022/1/25
		reachDist = np.linalg.norm(objPos - fingerCOM)
		
		placingGoal = self.sim.data.get_site_xpos('target0')
		placingGoal2 = self.sim.data.get_site_xpos('target1')
		self.goal = placingGoal
		assert (self.goal-placingGoal==0).any(), "goal does not match"

		placingDist = np.linalg.norm(objPos - placingGoal)
		placingDist2 = np.linalg.norm(objPos - placingGoal2)

		def reachReward():
			reachRew = -reachDist
			reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
			zRew = np.linalg.norm(np.linalg.norm(objPos[-1] - fingerCOM[-1]))


			if reachDistxy < 0.05:
				reachRew = -reachDist
			else:
				reachRew =  -reachDistxy - 2*zRew

			# incentive to close fingers when reachDist is small
			if reachDist < 0.05:
				reachRew = -reachDist + max(actions[-1],0)/50
			return reachRew , reachDist

		def pickCompletionCriteria():
			tolerance = 0.01
			return objPos[2] >= (heightTarget- tolerance)

		self.pickCompleted = pickCompletionCriteria()


		def objDropped():
			return (objPos[2] < (self.objHeight + 0.005)) and (reachDist > 0.02)
			# Object on the ground, far away from the goal, and from the gripper

		def orig_pickReward():
			hScale = 100
			if self.pickCompleted and not(objDropped()):
				return hScale*heightTarget
			elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)):
				return hScale* min(heightTarget, objPos[2])
			else:
				return 0
		

		def placeReward():
			c1 = 1000
			c2 = 0.01
			c3 = 0.001
			cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())

			if cond:
				placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
				placeRew = max(placeRew,0)
				return [placeRew , placingDist]
			else:
				return [0 , placingDist]

		def placeReward2():
			c1 = 2000
			c2 = 0.01
			c3 = 0.001
			cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped()) and (placingDist < 0.05)

			if cond:
				placeRew = 2000*(self.maxPlacingDist - placingDist2) + c1*(np.exp(-(placingDist2**2)/c2) + np.exp(-(placingDist2**2)/c3))
				placeRew = max(placeRew,0)
				return [placeRew , placingDist2]
			else:
				return [0 , placingDist2]

		def directPlaceReward(): 
			# give a reward for the robot to make object close
			if(objPos[2] > heightTarget-0.1):
				return 1000*(self.maxPlacingDist - placingDist)
			else:
				return 0

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		placeRew , placingDist = placeReward()
		placeRew2 , placingDist2 = placeReward2()
		direct_place_reward = directPlaceReward()
		assert ((placeRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + placeRew + placeRew2 + direct_place_reward

		return reward


	def compute_reward_v1(self, achieved_goal, goal, info):
    		
		actions = self.current_action
		
		objPos = self.sim.data.get_site_xpos('object0').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		# heightTarget = self.lift_height + self.objHeight
		heightTarget = self.sim.data.get_site_xpos('target0')[-1] - 0.05 # 2022/1/25
		reachDist = np.linalg.norm(objPos - fingerCOM)
		
		placingGoal = self.sim.data.get_site_xpos('target0')
		self.goal = placingGoal
		assert (self.goal-placingGoal==0).any(), "goal does not match"

		placingDist = np.linalg.norm(objPos - placingGoal)

		def reachReward():
			reachRew = -reachDist
			reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
			zRew = np.linalg.norm(np.linalg.norm(objPos[-1] - fingerCOM[-1]))


			if reachDistxy < 0.05:
				reachRew = -reachDist
			else:
				reachRew =  -reachDistxy - 2*zRew

			# incentive to close fingers when reachDist is small
			if reachDist < 0.05:
				reachRew = -reachDist + max(actions[-1],0)/50
			return reachRew , reachDist

		def pickCompletionCriteria():
			tolerance = 0.01
			return objPos[2] >= (heightTarget- tolerance)

		self.pickCompleted = pickCompletionCriteria()


		def objDropped():
			return (objPos[2] < (self.objHeight + 0.005)) and (reachDist > 0.02)
			# Object on the ground, far away from the goal, and from the gripper

		def orig_pickReward():
			hScale = 100
			if self.pickCompleted and not(objDropped()):
				return hScale*heightTarget
			elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)):
				return hScale* min(heightTarget, objPos[2])
			else:
				return 0
		

		def placeReward():
			c1 = 1000
			c2 = 0.01
			c3 = 0.001
			cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())

			if cond:
				placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
				placeRew = max(placeRew,0)
				return [placeRew , placingDist]
			else:
				return [0 , placingDist]
		
		def directPlaceReward(): 
			# give a reward for the robot to make object close
			if(objPos[2] > heightTarget-0.1):
				return 1000*(self.maxPlacingDist - placingDist)
			else:
				return 0

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		placeRew , placingDist = placeReward()
		direct_place_reward = directPlaceReward()# added in 2022/1/23
		assert ((placeRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + placeRew + direct_place_reward

		return reward

	def _get_state_obs(self):
    	# 1
		grasp_pos = self.sim.data.get_site_xpos("grasp")

		# 2
		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8
		
		# 3
		object_pos = self.sim.data.get_site_xpos('object0')
		
		# 4
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')

		# 5
		goal_pos = self.sim.data.get_site_xpos('target0')
		
		# concat
		state = np.concatenate([grasp_pos,[gripper_angle], object_pos, object_qpos, goal_pos  ])

		return state

	def _reset_sim(self):
		self.over_obj = False
		self.lifted = False # reset stage flag
		self.placed = False # reset stage flag
		self.over_goal = False

		return BaseEnv._reset_sim(self)

	def _set_action(self, action):
		assert action.shape == (4,)

		if self.flipbit:
			action[3] = 0
			self.flipbit = 0
		else:
			action[:3] = np.zeros(3)
			self.flipbit = 1
			
		self.current_action = action # store current_action

		BaseEnv._set_action(self, action)


	def _get_achieved_goal(self):
		"""
		Get the position of the target pos.
		"""
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _sample_object_pos(self):
		"""
		Sample the initial position of the object
		"""
		object_xpos = np.array([1.43, 0.29, 0.575]) # to align with real
		object_xpos = self.center_of_table.copy() - np.array([0.35, 0, 0.05])
		object_xpos[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		object_xpos[1] += self.np_random.uniform(-0.1, 0.1, size=1)
	
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('object0:joint', object_qpos)
		
		self.obj_init_pos = object_xpos # store this position, used in the reward
		self.objHeight = self.obj_init_pos[2]
		
		self.maxPlacingDist = np.linalg.norm(self.obj_init_pos - self.goal)


	def _sample_goal(self, new=True):
		"""
		Sample the position of the shelf, and the goal is bound to the shelf.
		"""
		# shelf_qpos = self.sim.data.get_joint_qpos('shelf:joint') 
		# shelf_xpos = shelf_qpos[:3]
		# shelf_quat = shelf_qpos[-4:]
		
		# if new:
			# randomize the position of the shelf
			# shelf_xpos[0] += self.env_randomness_scale * self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
			# shelf_xpos[1] += self.np_random.uniform(-0.01 - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
			# shelf_xpos[1] = self.sim.data.get_site_xpos('object0')[1]
			# shelf_xpos[1] += self.env_randomness_scale * self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)

			# shelf_qpos[:3] = shelf_xpos
			# shelf_qpos[-4:] = shelf_quat

			# self.sim.data.set_joint_qpos('shelf:joint', shelf_qpos)
		# else:
			# pass
		
		
		self.lift_height = 0.15


		goal = self.sim.data.get_site_xpos('target0')
		
		self.heightTarget = goal[2]

		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		"""
		Sample the initial position of arm
		"""
		gripper_target = np.array([1.28, .295, 0.71])
		gripper_target[0] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[1] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[2] += self.np_random.uniform(-0.02, 0.02, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)