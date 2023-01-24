import numpy as np
import os
import env.robot.reward_utils as reward_utils
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class ShelfPlacingEnv(BaseEnv, utils.EzPickle):
	"""
	Place the object on the shelf
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False, env_randomness_scale=1.0):
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
		self.distance_threshold = 0.1
		
		utils.EzPickle.__init__(self)

	def _gripper_caging_reward(self,
							action,
							obj_pos,
							obj_radius,
							pad_success_thresh,
							object_reach_radius,
							xz_thresh,
							desired_gripper_effort=1.0,
							high_density=False,
							medium_density=False):
		"""Reward for agent grasping obj
			Args:
				action(np.ndarray): (4,) array representing the action
					delta(x), delta(y), delta(z), gripper_effort
				obj_pos(np.ndarray): (3,) array representing the obj x,y,z
				obj_radius(float):radius of object's bounding sphere
				pad_success_thresh(float): successful distance of gripper_pad
					to object
				object_reach_radius(float): successful distance of gripper center
					to the object.
				xz_thresh(float): successful distance of gripper in x_z axis to the
					object. Y axis not included since the caging function handles
						successful grasping in the Y axis.
		"""
		if high_density and medium_density:
			raise ValueError("Can only be either high_density or medium_density")
		# MARK: Left-right gripper information for caging reward----------------
		right_pad = self.sim.data.get_body_xpos('right_hand').copy()
		left_pad = self.sim.data.get_body_xpos('left_hand').copy()
		# left_pad = self.get_body_com('leftpad')
		# right_pad = self.get_body_com('rightpad')

		# get current positions of left and right pads (Y axis)
		pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
		# compare *current* pad positions with *current* obj position (Y axis)
		pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
		# compare *current* pad positions with *initial* obj position (Y axis)
		pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

		# Compute the left/right caging rewards. This is crucial for success,
		# yet counterintuitive mathematically because we invented it
		# accidentally.
		#
		# Before touching the object, `pad_to_obj_lr` ("x") is always separated
		# from `caging_lr_margin` ("the margin") by some small number,
		# `pad_success_thresh`.
		#
		# When far away from the object:
		#       x = margin + pad_success_thresh
		#       --> Thus x is outside the margin, yielding very small reward.
		#           Here, any variation in the reward is due to the fact that
		#           the margin itself is shifting.
		# When near the object (within pad_success_thresh):
		#       x = pad_success_thresh - margin
		#       --> Thus x is well within the margin. As long as x > obj_radius,
		#           it will also be within the bounds, yielding maximum reward.
		#           Here, any variation in the reward is due to the gripper
		#           moving *too close* to the object (i.e, blowing past the
		#           obj_radius bound).
		#
		# Therefore, before touching the object, this is very nearly a binary
		# reward -- if the gripper is between obj_radius and pad_success_thresh,
		# it gets maximum reward. Otherwise, the reward very quickly falls off.
		#
		# After grasping the object and moving it away from initial position,
		# x remains (mostly) constant while the margin grows considerably. This
		# penalizes the agent if it moves *back* toward `obj_init_pos`, but
		# offers no encouragement for leaving that position in the first place.
		# That part is left to the reward functions of individual environments.
		caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
		caging_lr = [reward_utils.tolerance(
			pad_to_obj_lr[i],  # "x" in the description above
			bounds=(obj_radius, pad_success_thresh),
			margin=caging_lr_margin[i],  # "margin" in the description above
			sigmoid='long_tail',
		) for i in range(2)]
		caging_y = reward_utils.hamacher_product(*caging_lr)

		# MARK: X-Z gripper information for caging reward-----------------------
		tcp = self.tcp_center
		xz = [0, 2]

		# Compared to the caging_y reward, caging_xz is simple. The margin is
		# constant (something in the 0.3 to 0.5 range) and x shrinks as the
		# gripper moves towards the object. After picking up the object, the
		# reward is maximized and changes very little
		caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.initial_gripper_xpos[xz])
		caging_xz_margin -= xz_thresh
		caging_xz = reward_utils.tolerance(
			np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
			bounds=(0, xz_thresh),
			margin=caging_xz_margin,  # "margin" in the description above
			sigmoid='long_tail',
		)

		# MARK: Closed-extent gripper information for caging reward-------------
		gripper_closed = min(max(0, action[-1]), desired_gripper_effort) \
							/ desired_gripper_effort

		# MARK: Combine components----------------------------------------------
		caging = reward_utils.hamacher_product(caging_y, caging_xz)
		gripping = gripper_closed if caging > 0.97 else 0.
		caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

		if high_density:
			caging_and_gripping = (caging_and_gripping + caging) / 2
		if medium_density:
			tcp = self.tcp_center
			tcp_to_obj = np.linalg.norm(obj_pos - tcp)
			tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.initial_gripper_xpos)
			# Compute reach reward
			# - We subtract `object_reach_radius` from the margin so that the
			#   reward always starts with a value of 0.1
			reach_margin = abs(tcp_to_obj_init - object_reach_radius)
			reach = reward_utils.tolerance(
				tcp_to_obj,
				bounds=(0, object_reach_radius),
				margin=reach_margin,
				sigmoid='long_tail',
			)
			caging_and_gripping = (caging_and_gripping + reach) / 2

		return caging_and_gripping


	def compute_reward_shelfplacing_v1(self, achieved_goal, goal, info):
		_TARGET_RADIUS = 0.05
		tcp_opened = self.current_action[3] # can this be replaced by the gripper angle?

		# -------
		action = self.current_action
		obs = achieved_goal

		self.tcp_center = self.sim.data.get_site_xpos('grasp').copy()
		tcp = self.tcp_center # position of grasper
		obj = self.sim.data.get_site_xpos('object0').copy() # position of object
		
		# target = goal.copy() FIXME: why goal and get_site_xpos are different?
		target = self.sim.data.get_site_xpos('target0').copy()	
		

		# ---------

		
		obj_to_target = np.linalg.norm(obj - target)
		tcp_to_obj = np.linalg.norm(obj - tcp)
		in_place_margin = np.linalg.norm(self.obj_init_pos - target)
		
		in_place = reward_utils.tolerance(obj_to_target,
									bounds=(0, _TARGET_RADIUS),
									margin=in_place_margin,
									sigmoid='long_tail',)

		object_grasped = self._gripper_caging_reward(action=action,
														obj_pos=obj,
														obj_radius=0.02,
														pad_success_thresh=0.05,
														object_reach_radius=0.01,
														xz_thresh=0.01,
														high_density=False)
		
		reward = reward_utils.hamacher_product(object_grasped, in_place)

		# action penalty
		reward = -0.1*np.square(self._pos_ctrl_magnitude)
		
		if (0.0 < obj[2] < 0.24 and \
				(target[0]-0.15 < obj[0] < target[0]+0.15) and \
				((target[1] - 3*_TARGET_RADIUS) < obj[1] < target[1])):
			z_scaling = (0.24 - obj[2])/0.24
			y_scaling = (obj[1] - (target[1] - 3*_TARGET_RADIUS)) / (3*_TARGET_RADIUS)
			bound_loss = reward_utils.hamacher_product(y_scaling, z_scaling)
			in_place = np.clip(in_place - bound_loss, 0.0, 1.0)

		if ((0.0 < obj[2] < 0.24) and \
				(target[0]-0.15 < obj[0] < target[0]+0.15) and \
				(obj[1] > target[1])):
			in_place = 0.0

		if tcp_to_obj < 0.025 and (tcp_opened > 0) and \
				(obj[2] - 0.01 > self.obj_init_pos[2]):
			reward += 1. + 5. * in_place

		if obj_to_target < _TARGET_RADIUS:
			reward += 10. # or =100

		return reward

	def compute_reward_shelfplacing_v2(self, achieved_goal, goal, info):
		
		actions = self.current_action

		objPos = achieved_goal

		rightFinger, leftFinger = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		fingerCOM  =  (rightFinger + leftFinger)/2

		heightTarget = self.sim.data.get_site_xpos('target0')[2]
		placingGoal = self.sim.data.get_site_xpos('target0')

		reachDist = np.linalg.norm(objPos - fingerCOM)

		placingDist = np.linalg.norm(objPos - placingGoal)


		def reachReward():
			reachRew = -reachDist
			reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
			zRew = np.linalg.norm(fingerCOM[-1] - self.init_finger_xpos[-1])

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
			return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
			# Object on the ground, far away from the goal, and from the gripper
			# Can tweak the margin limits

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

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		placeRew , placingDist = placeReward()
		assert ((placeRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + placeRew

		# [reward, reachRew, reachDist, pickRew, placeRew, placingDist]
		return reward


	def compute_reward_shelfplacing_v3(self, achieved_goal, goal, info):
		
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

	def compute_reward(self, achieved_goal, goal, info):
		return self.compute_reward_shelfplacing_v3(achieved_goal, goal, info)

	def _get_state_obs(self):
    	# 1
		grasp_pos = self.sim.data.get_site_xpos("grasp")

		# 2
		finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
		
		# 3
		object_pos = self.sim.data.get_site_xpos('object0')
		
		# 4
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')

		# 5
		goal_pos = self.sim.data.get_site_xpos('target0')
		
		# concat
		state = np.concatenate([grasp_pos,[gripper_distance_apart], object_pos, object_qpos, goal_pos  ])

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
		
		BaseEnv._set_action(self, action)
		self.current_action = action # store current_action

	def _get_achieved_goal(self):
		"""
		Get the position of the target pos.
		"""
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _sample_object_pos(self):
		"""
		Sample the initial position of the object
		"""
		
		object_xpos = self.center_of_table.copy() - np.array([0.30, 0, 0])
		object_xpos[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		object_xpos[1] += 0.1*self.np_random.uniform(-0.1, 0.1, size=1)
		object_xpos[2] += -0.012
	
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('object0:joint', object_qpos)
		
		self.obj_init_pos = object_xpos # store this position, used in the reward
		self.objHeight = self.obj_init_pos[2]
		
		
		self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) \
							- np.array(self._target_pos)) + self.heightTarget


	def _sample_goal(self, new=True):
		"""
		Sample the position of the shelf, and the goal is bound to the shelf.
		"""
		shelf_qpos = self.sim.data.get_joint_qpos('shelf:joint') 
		shelf_xpos = shelf_qpos[:3]
		shelf_quat = shelf_qpos[-4:]
		
		if new:
			# randomize the position of the shelf
			shelf_xpos[1] = self.sim.data.get_site_xpos('object0')[1]
			shelf_xpos[0] = self.sim.data.get_site_xpos('object0')[0] + 0.25
			shelf_xpos[0] += self.env_randomness_scale * self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
			shelf_xpos[1] += self.env_randomness_scale * self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)

			shelf_qpos[:3] = shelf_xpos
			shelf_qpos[-4:] = shelf_quat

			self.sim.data.set_joint_qpos('shelf:joint', shelf_qpos)
		else:
			pass
		
		
		self.lift_height = 0.15


		goal = self.sim.data.get_site_xpos('target0')
		self._target_pos = self.sim.data.get_site_xpos('target0')
		
		self.heightTarget = goal[2]

		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		"""
		Sample the initial position of arm
		"""
		gripper_target = np.array([1.2561169, 0.3, 0.69603332])
		gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(-0.05, 0.05, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)