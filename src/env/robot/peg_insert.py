import numpy as np
import os
import env.robot.reward_utils as reward_utils
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class peginsertEnv(BaseEnv, utils.EzPickle):
	"""
	Insert into the box
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		self.sample_large = 1
		
		self.box_init_pos = None

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
		
		self.state_dim = (31,)
		self.flipbit = 1
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
		pad_to_objinit_lr = np.abs(pad_y_lr - self.peg_init_pos[1])

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
		caging_xz_margin = np.linalg.norm(self.peg_init_pos[xz] - self.initial_gripper_xpos[xz])
		caging_xz_margin -= xz_thresh
		# print("caging:", caging_xz_margin)
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

	
	def compute_reward_v1(self, achieved_goal, goal, info):
    		
		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8
		actions = self.current_action

		objPos = self.sim.data.get_site_xpos('pegGrasp').copy()

		self.objHeight = objPos[2]
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		heightTarget = self.sim.data.get_site_xpos('goal')[-1]
		reachDist = np.linalg.norm( objPos - fingerCOM )
		
		placingDist = np.linalg.norm(objPos - goal)

		def reachReward():
			reachRew = -reachDist
			reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
			zRew = np.linalg.norm(np.linalg.norm(objPos[-1] - fingerCOM[-1]))


			if reachDistxy < 0.05:
				reachRew = -reachDist
			else:
				reachRew =  -reachDist - 2*zRew

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
				return 10*(self.maxPlacingDist - placingDist)
			else:
				return 0

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		placeRew , placingDist = placeReward()
		direct_place_reward = directPlaceReward() 
		assert ((placeRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + placeRew + direct_place_reward

		return reward

	def compute_reward_v2(self, achieved_goal, goal, info):
		self.TARGET_RADIUS = 0.05
		self.tcp_center = self.sim.data.get_site_xpos('grasp').copy()

		tcp = self.tcp_center
		obj = self.sim.data.get_body_xpos('peg').copy()
		obj_head = self.sim.data.get_site_xpos('pegHead').copy()
		hole_pos = self.sim.data.get_site_xpos('hole').copy()
		tcp_opened = - self.current_action[3]

		target = goal
		tcp_to_obj = np.linalg.norm(obj - tcp)
		scale = np.array([1., 2., 2.])
		#  force agent to pick up object then insert
		obj_to_target = np.linalg.norm((obj_head - target) * scale)

		in_place_margin = np.linalg.norm((self.peg_init_pos - target) * scale)
		in_place = reward_utils.tolerance(obj_to_target,
									bounds=(0, self.TARGET_RADIUS),
									margin=in_place_margin,
									sigmoid='long_tail',)
		ip_orig = in_place
		brc_col_box_1 = self.sim.data.get_site_xpos('bottom_right_corner_collision_box_1')
		tlc_col_box_1 = self.sim.data.get_site_xpos('top_left_corner_collision_box_1')

		brc_col_box_2 = self.sim.data.get_site_xpos('bottom_right_corner_collision_box_2')
		tlc_col_box_2 = self.sim.data.get_site_xpos('top_left_corner_collision_box_2')
		collision_box_bottom_1 = reward_utils.rect_prism_tolerance(curr=obj_head,
																	one=tlc_col_box_1,
																	zero=brc_col_box_1)
		collision_box_bottom_2 = reward_utils.rect_prism_tolerance(curr=obj_head,
																	one=tlc_col_box_2,
																	zero=brc_col_box_2)
		collision_boxes = reward_utils.hamacher_product(collision_box_bottom_2,
														collision_box_bottom_1)
		in_place = reward_utils.hamacher_product(in_place,
													collision_boxes)

		pad_success_margin = 0.03
		object_reach_radius=0.01
		x_z_margin = 0.005
		obj_radius = 0.0075

		object_grasped = self._gripper_caging_reward(self.current_action,
														obj,
														object_reach_radius=object_reach_radius,
														obj_radius=obj_radius,
														pad_success_thresh=pad_success_margin,
														xz_thresh=x_z_margin,
														high_density=True)
		if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.peg_init_pos[2]):
			object_grasped = 1.
		in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
																	in_place)
		reward = in_place_and_object_grasped

		if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.peg_init_pos[2]):
			reward += 1. + 5 * in_place

		if obj_to_target <= 0.07:
			reward = 10.


		return reward
	
	def compute_reward(self, achieved_goal, goal, info):
    		
		actions = self.current_action
		
		objPos = self.sim.data.get_site_xpos('pegGrasp').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		# heightTarget = self.lift_height + self.objHeight
		heightTarget = self.sim.data.get_site_xpos('goal')[-1] - 0.05 # 2022/1/25
		reachDist = np.linalg.norm(objPos - fingerCOM)
		
		placingGoal = self.sim.data.get_site_xpos('goal')
		self.goal = placingGoal
		
		self.objHeight = objPos[2]

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
		peg_grasp = self.sim.data.get_site_xpos('pegGrasp')
		peg_head = self.sim.data.get_site_xpos('pegHead')
		peg_end = self.sim.data.get_site_xpos('pegEnd')
		

		# 4
		# hole and goal are different
		hole_pos = self.sim.data.get_site_xpos('hole').copy()
		goal_pos = self.goal

		# 5
		box_bottom_right_corner_1 = self.sim.data.get_site_xpos('bottom_right_corner_collision_box_1')
		box_bottom_right_corner_2 = self.sim.data.get_site_xpos('bottom_right_corner_collision_box_2')
		box_top_left_corner_1 = self.sim.data.get_site_xpos('top_left_corner_collision_box_1')
		box_top_left_corner_2 = self.sim.data.get_site_xpos('top_left_corner_collision_box_2')

		# concat
		state = np.concatenate([grasp_pos,[gripper_angle],peg_head, peg_grasp,peg_end , hole_pos, goal_pos, 
						box_bottom_right_corner_1, box_bottom_right_corner_2, box_top_left_corner_1, box_top_left_corner_2])

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
		return np.squeeze(self.sim.data.get_site_xpos('pegHead').copy())

	def _sample_object_pos(self):
		"""
		Sample the initial position of the peg
		"""
		object_xpos = np.array([1.33, 0.29, 0.565]) # to align with real
		object_xpos[0] += self.np_random.uniform(-0.03, 0.03, size=1)
		object_xpos[1] += self.np_random.uniform(-0.07, 0.07, size=1)
	
		object_qpos = self.sim.data.get_joint_qpos('peg:joint')
		object_quat = object_qpos[-4:]

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('peg:joint', object_qpos)


		self.peg_init_pos = object_xpos.copy()
		self.peg_height = object_xpos[-1]

		
		
		self.initial_gripper_xpos = self.sim.data.get_site_xpos('grasp').copy()
		self.maxPlacingDist = np.linalg.norm(self.peg_init_pos - self.goal)

	def _sample_goal(self, new=True):

		# Randomly sample the position of the box
		box_pos = self.sim.data.get_body_xpos("box")
		
		if self.box_init_pos is None:
			self.box_init_pos = box_pos.copy()
		else:
			box_pos[0] = self.box_init_pos[0] + self.np_random.uniform(-0.05, 0.05, size=1)
			box_pos[1] = self.box_init_pos[1] + 0.1*self.np_random.uniform(-0.1, 0.1, size=1)
			box_pos[2] = self.box_init_pos[2]
		
			
		goal = self.sim.data.get_site_xpos('goal').copy()

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