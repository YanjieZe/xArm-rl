import numpy as np
import os
import env.robot.reward_utils as reward_utils
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class HammerPickEnv(BaseEnv, utils.EzPickle):
	"""
	Place the object on the shelf
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		self.sample_large = 1
		self.state_dim = (13,)
		self.statefull_dim = self.state_dim
		# different rotation of the gripper, origin is [0,1,0,0]
		gripper_rotation = [0,1,1,0]
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
			has_object=True,
			gripper_rotation=gripper_rotation
		)
		
		self.flipbit = 1

		self.distance_threshold = 0.05
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


	def compute_reward_v2(self, achieved_goal, goal, info):
		
		hand = self.sim.data.get_site_xpos("grasp")
		hammer = self.sim.data.get_body_xpos("hammerbody")
		hammer_head = hammer + np.array([.16, .06, .0])
		# `self._gripper_caging_reward` assumes that the target object can be
		# approximated as a sphere. This is not true for the hammer handle, so
		# to avoid re-writing the `self._gripper_caging_reward` we pass in a
		# modified hammer position.
		# This modified position's X value will perfect match the hand's X value
		# as long as it's within a certain threshold
		hammer_threshed = hammer.copy()
		HAMMER_HANDLE_LENGTH = 0.14
		threshold = HAMMER_HANDLE_LENGTH / 2.0
		if abs(hammer[0] - hand[0]) < threshold:
			hammer_threshed[0] = hand[0]

		def _reward_quat(obs):
			# Ideal laid-down wrench has quat [1, 0, 0, 0]
			# Rather than deal with an angle between quaternions, just approximate:
			ideal = np.array([1., 0., 0., 0.])
			error = np.linalg.norm(obs[7:11] - ideal)
			return max(1.0 - error / 0.4, 0.0)

		reward_quat = _reward_quat(obs)
		reward_grab = self._gripper_caging_reward(
			actions, hammer_threshed,
			object_reach_radius=0.01,
			obj_radius=0.015,
			pad_success_thresh=0.02,
			xz_thresh=0.01,
			high_density=True,
		)
		reward_in_place = SawyerHammerEnvV2._reward_pos(
			hammer_head,
			self._target_pos
		)

		reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
		# Override reward on success. We check that reward is above a threshold
		# because this env's success metric could be hacked easily
		success = self.data.get_joint_qpos('NailSlideJoint') > 0.09
		if success and reward > 5.:
			reward = 10.0

		return reward

	def compute_reward_v1(self, achieved_goal, goal, info):
	
		actions =  self.current_action
		
		hammerPos = self.sim.data.get_body_xpos('hammerbody')
		self.hammerHeight = hammerPos[-1]

		hammerHeadPos = self.sim.data.get_geom_xpos('hammerHead').copy()
		objPos = self.sim.data.get_site_xpos("nailHead")

		rightFinger, leftFinger = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		fingerCOM  =  (rightFinger + leftFinger)/2

		heightTarget = self.heightTarget

		hammerDist = np.linalg.norm(objPos - hammerHeadPos)

		

		screwDist = np.abs(objPos[1] - self._target_pos[1])
		reachDist = np.linalg.norm(hammerPos - fingerCOM)

		def reachReward():
			reachRew = -reachDist
			# incentive to close fingers when reachDist is small
			if reachDist < 0.05:
				reachRew = -reachDist + max(actions[-1],0)/50
			return reachRew , reachDist

		def pickCompletionCriteria():
			tolerance = 0.01
			if hammerPos[2] >= (heightTarget- tolerance):
				return True
			else:
				return False


		if pickCompletionCriteria():
			self.pickCompleted = True
		else:
			self.pickCompleted = False		


		def objDropped():
			return (hammerPos[2] < (self.hammerHeight + 0.005)) and (hammerDist >0.02) and (reachDist > 0.02)
			# Object on the ground, far away from the goal, and from the gripper
			# Can tweak the margin limits

		def orig_pickReward():
			hScale = 100

			if self.pickCompleted and not(objDropped()):
				return hScale*heightTarget
			elif (reachDist < 0.1) and (hammerPos[2]> (self.hammerHeight + 0.005)) :
				return hScale* min(heightTarget, hammerPos[2])
			else:
				return 0

		def hammerReward():
			c1 = 1000
			c2 = 0.01
			c3 = 0.001

			cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
			if cond:
				hammerRew = 1000*(self.maxHammerDist - hammerDist - screwDist) + c1*(np.exp(-((hammerDist+screwDist)**2)/c2) + np.exp(-((hammerDist+screwDist)**2)/c3))
				hammerRew = max(hammerRew,0)
				return [hammerRew , hammerDist, screwDist]
			else:
				return [0 , hammerDist, screwDist]

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		hammerRew , hammerDist, screwDist = hammerReward()
		assert ((hammerRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + hammerRew

		return reward

	def compute_reward(self, achieved_goal, goal, info):
    	# metaworld reward v1 or v2
		# v1 's scale is quiet large
		return self.compute_reward_v1(achieved_goal, goal, info)

	def _get_state_obs(self):
    	# 1
		grasp_pos = self.sim.data.get_site_xpos("grasp")

		# 2
		finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
		
		# 3
		hammer_pos = self.sim.data.get_body_xpos('hammerbody')
		
		# 4
		nail_pos = self.sim.data.get_site_xpos('nailHead')

		# 5
		goal_pos = self.sim.data.get_site_xpos('goal')
		
		# concat
		state = np.concatenate([grasp_pos,[gripper_distance_apart], hammer_pos, nail_pos, goal_pos  ])
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
		return self.sim.data.get_site_xpos('nail_head').copy()
		# return self.sim.data.get_joint_qpos('NailSlideJoint').copy()

	def _sample_object_pos(self):
		"""
		Sample the initial position of the object
		"""

		object_xpos = self.center_of_table.copy() - np.array([0.25, 0, 0.08])
		object_xpos[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		object_xpos[1] += self.np_random.uniform(-0.1, 0.1, size=1)
	
		object_qpos = self.sim.data.get_joint_qpos('hammerbody:joint')
		object_quat = object_qpos[-4:]
		
		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('hammerbody:joint', object_qpos)
		
		hammerbody_xpos = self.sim.data.get_body_xpos('hammerbody') 
		hammerhead_xpos = self.sim.data.get_geom_xpos('hammerHead').copy()

		grasp_pos = self.sim.data.get_site_xpos('grasp')

		hammerbody_xpos[1] = grasp_pos[1] # align the y axis

		box_xpos = self.sim.data.get_body_xpos('box')

		box_xpos[0] = hammerbody_xpos[0]  + 0.3 +self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
		# box_xpos[1] = hammerhead_xpos[1] - 0.2 # align the y axis of the box and the hammer
		box_xpos[1] = hammerhead_xpos[1]


	def _sample_goal(self, new=True):
		"""
		Sample the position of the shelf, and the goal is bound to the shelf.
		"""
		hammerbody_xpos = self.sim.data.get_body_xpos('hammerbody') 
		hammerhead_xpos = self.sim.data.get_geom_xpos('hammerHead').copy()

		grasp_pos = self.sim.data.get_site_xpos('grasp')

		hammerbody_xpos[1] = grasp_pos[1] # align the y axis

		box_xpos = self.sim.data.get_body_xpos('box')

		if new:
			# randomize the position
			hammerbody_xpos[0] += self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
			hammerbody_xpos[1] += self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)

			box_xpos[0] = hammerbody_xpos[0]  + 0.3 +self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
			# box_xpos[1] = hammerhead_xpos[1] - 0.2 # align the y axis of the box and the hammer
			box_xpos[1] = hammerhead_xpos[1] + 0.9

		else:
			pass
		
		
		self.lift_height = 0.15

		objPos = self.sim.data.get_site_xpos("nailHead")
		hammerHeadPos = self.sim.data.get_geom_xpos('hammerHead').copy()
		self.maxHammerDist  = np.linalg.norm(objPos - hammerHeadPos)
		
		goal = self.sim.data.get_site_xpos('goal')
		# goal = 0.09
		self._target_pos = goal
		
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