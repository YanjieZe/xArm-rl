import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class PickPlaceEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):

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
		self.state_dim = (26,) if self.use_xyz else (20,)
		self.flipbit = 1
		utils.EzPickle.__init__(self)

	def compute_reward_v1(self, achieved_goal, goal, info):
		eef_pos = self.sim.data.get_site_xpos('grasp').copy()
		object_pos = self.sim.data.get_site_xpos('object0').copy()
		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8

		goal_pos = goal.copy()


		d_eef_obj = self.goal_distance(eef_pos, object_pos, self.use_xyz)
		d_eef_obj_xy = self.goal_distance(eef_pos, object_pos, use_xyz  = False)
		d_obj_goal_xy = self.goal_distance(object_pos, goal_pos, use_xyz=False)
		d_obj_goal_xyz = self.goal_distance(object_pos, goal_pos, use_xyz=True)
		eef_z = eef_pos[2] - self.center_of_table.copy()[2]
		obj_z = object_pos[2] - self.center_of_table.copy()[2]

		reward = -0.1*np.square(self._pos_ctrl_magnitude) # action penalty
		
		if not self.over_obj :
			reward += -2 * d_eef_obj_xy # penalty for not reaching object
			if d_eef_obj_xy <= 0.05 and not self.over_obj:
				self.over_obj = True
		elif not self.lifted:
			reward += 6*min(max(obj_z, 0), self.lift_height)  - 3*self.goal_distance(eef_pos, object_pos, self.use_xyz)
			if obj_z > self.lift_height and self.goal_distance(eef_pos, object_pos, self.use_xyz) <= 0.05 and not self.lifted:
				self.lifted = True
		elif not self.over_goal:
			reward += 2 -3*d_obj_goal_xy + 6*min(max(obj_z, 0), self.lift_height)
			if d_obj_goal_xy < 0.06 and not self.over_goal:
				self.over_goal = True
		elif not self.placed:
			reward += 10 - 20*d_obj_goal_xyz - 5 * gripper_angle
			if d_obj_goal_xyz < 0.05 and not self.placed:
				self.placed = True
		else :
			reward += 100*min(max(eef_z, 0), self.lift_height)

		return reward

	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal

		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8

		obj_pos = self.sim.data.get_site_xpos('object0')
		obj_rot = self.sim.data.get_joint_qpos('object0:joint')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('object0') * dt
		obj_velr = self.sim.data.get_site_xvelr('object0') * dt

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]
			obj_pos = obj_pos[:2]
			obj_velp = obj_velp[:2]
			obj_velr = obj_velr[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			self.goal_distance(obj_pos, goal_pos, self.use_xyz),
			self.goal_distance(eef_pos, obj_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, obj_pos, obj_rot, obj_velp, obj_velr, values
		], axis=0)

	def compute_reward(self, achieved_goal, goal, info):
    		
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

		self.pick_and_to_drop = False
		def dropReward():
			if placingDist < 0.03:
				# incentive to open fingers when placeDist is small
				self.pick_and_to_drop = True
				return 10000  - 1000 * max(actions[-1],0)/50
			elif self.pick_and_to_drop:
				return 20000
			else:
				return 0

		def directPlaceReward(): 
			# give a reward for the robot to make object close
			if(objPos[2] > heightTarget-0.1):
				return 1000*(self.maxPlacingDist - placingDist)
			else:
				return 0

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		placeRew , placingDist = placeReward()
		dropRew = dropReward()
		direct_place_reward = directPlaceReward()# added in 2022/1/23
		assert ((placeRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + placeRew + direct_place_reward + dropRew

		return reward



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
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _sample_object_pos(self):

		object_xpos = np.array([1.33, 0.22, 0.565]) # to align with real
		object_xpos[0] += self.np_random.uniform(-0.03, 0.03, size=1)
		object_xpos[1] += self.np_random.uniform(-0.05, 0.05, size=1)
	
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]


		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3]
		object_qpos[-4:] = object_quat
		
		
		self.sim.data.set_joint_qpos('object0:joint', object_qpos)

		self.obj_init_pos = object_xpos # store this position, used in the reward
		self.objHeight = self.obj_init_pos[2]
		self.maxPlacingDist = np.linalg.norm(self.obj_init_pos - self.goal)

	def _sample_goal(self, new=True):
		object_qpos = self.sim.data.get_joint_qpos('box_hole:joint')
		object_quat = object_qpos[-4:]

		if new:
			goal = object_qpos[:3].copy()
			goal[0] += self.np_random.uniform(-0.01, 0.01 , size=1)
			goal[1] += self.np_random.uniform(-0.01, 0.01, size=1)
		else:
			goal = object_qpos[:3].copy()
		

		object_qpos[:3] = goal[:3].copy()
		object_qpos[-4:] = object_quat

		self.sim.data.set_joint_qpos('box_hole:joint', object_qpos)
		self.lift_height = 0.15

		goal = self.sim.data.get_site_xpos('target0').copy()
		
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.28, .295, 0.71])
		gripper_target[0] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[1] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[2] += self.np_random.uniform(-0.02, 0.02, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)
