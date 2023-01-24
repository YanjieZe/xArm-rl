import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path
import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class ReachWallEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84,
				 use_xyz=False, render=False):
		self.sample_large = 1
		BaseEnv.__init__(self,
						 get_full_asset_path(xml_path),
						 n_substeps=n_substeps,
						 observation_type=observation_type,
						 reward_type=reward_type,
						 image_size=image_size,
						 reset_free=False,
						 cameras=cameras,
						 render=render,
						 use_xyz=use_xyz
						 )
		self.state_dim = (17,) if self.use_xyz else (12,)
		self.reach1 = False
		utils.EzPickle.__init__(self)

	def reach_reward(self, achieved_goal, goal, TOLL, done=False):
		if done:
			return 1.0
		rightFinger, leftFinger = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		fingerCOM = (rightFinger + leftFinger) / 2

		reachDist = np.linalg.norm(achieved_goal - goal)

		reachDist_xy = np.linalg.norm(achieved_goal[:-1] - goal[:-1])
		# zRew = np.linalg.norm(fingerCOM[:-1] - self.init_finger_xpos[-1])

		if reachDist_xy > TOLL:
			reachRew = -reachDist
		else:
			reachRew = 1.0
		return reachRew


	def compute_reward(self, achieved_goal, goal, info):
		"""d = self.goal_distance(achieved_goal, goal, self.use_xyz)

		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			#return np.around(d, 4)
			return np.around(-3 * d - 0.5 * np.square(self._pos_ctrl_magnitude), 4)
		#return self.reach_reward(achieved_goal, goal, info)"""
		_TARGER_RADIUS = 0.05
		cum_rew = 0
		rightFinger, leftFinger = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		fingerCOM = (rightFinger + leftFinger) / 2

		mid1 = self.sim.data.get_site_xpos('wall_left')
		mid2 = self.sim.data.get_site_xpos('wall_right')

		if np.linalg.norm(fingerCOM - mid1) > np.linalg.norm(fingerCOM - mid2):
			mid = mid2
		else:
			mid = mid1

		temp = self.reach_reward(fingerCOM, mid, _TARGER_RADIUS, done=self.reach1)
		if temp == 1.0:
			self.reach1 = True

		cum_rew += temp
		cum_rew += self.reach_reward(goal, fingerCOM, _TARGER_RADIUS)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return np.around(cum_rew, 4)

	def _get_state_obs(self):
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')
		wall_pos_left = self.sim.data.get_site_xpos('wall_left')
		wall_pos_right = self.sim.data.get_site_xpos('wall_right')

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]
			wall_pos_left = wall_pos_left[:2]
			wall_pos_right = wall_pos_right[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, values, wall_pos_left, wall_pos_right
		], axis=0)

	def _get_achieved_goal(self):
		return self.sim.data.get_site_xpos('grasp').copy()

	def _sample_goal(self, new=True):
		site_id = self.sim.model.site_name2id('target0')

		#new = False
		if new:
			goal = np.array([1.75, 0.3, 0.58])
			goal[0] += self.np_random.uniform(-0.05 - 0.05 * self.sample_large, 0.05 + 0.05 * self.sample_large, size=1)
			goal[1] += self.np_random.uniform(-0.1 - 0.1 * self.sample_large, 0.1 + 0.1 * self.sample_large, size=1)
			#goal = self.center_of_table.copy() + np.array([0.2, 0, 0])
			#goal[0] += self.np_random.uniform(-0.05, -0.19, size=1)
			#goal[1] += self.np_random.uniform(-0.2, 0.2, size=1)
		else:
			goal = self.sim.data.get_site_xpos('target0')

		self.sim.model.site_pos[site_id] = goal
		self.sim.forward()

		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.2561169, 0.3, 0.58])# 0.58603332])
		gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)

		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(-0.05, 0.1, size=1)
		self.reach1 = False
		BaseEnv._sample_initial_pos(self, gripper_target)


class ReachMovingTargetEnv(ReachWallEnv):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_velocity()

	def set_velocity(self):
		self.curr_vel = 0.0025 * np.ones(2)

	def _sample_goal(self):
		self.set_velocity()
		return ReachEnv._sample_goal(self)

	def _step_callback(self):
		self.set_goal()

	def set_goal(self):
		curr_goal = self.goal

		if (curr_goal[0] >= 1.4 and self.curr_vel[0] > 0) \
				or curr_goal[0] <= 1.2 and self.curr_vel[0] < 0:
			self.curr_vel[0] = -1 * self.curr_vel[0]
		if (curr_goal[1] >= 0.2 and self.curr_vel[1] > 0) \
				or curr_goal[1] <= -0.2 and self.curr_vel[1] < 0:
			self.curr_vel[1] = -1 * self.curr_vel[1]
		self.goal[0] += self.curr_vel[0]
		self.goal[1] += self.curr_vel[1]
