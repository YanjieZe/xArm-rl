import numpy as np 
import os
import math
from numpy.core.defchararray import join
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path

class DrawerCloseEnv(BaseEnv, utils.EzPickle):
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
        self.state_dim = (22,) if self.use_xyz else (16,)
        utils.EzPickle.__init__(self)

 
    # def compute_reward(self, achieved_goal, goal, info):
    #     eef_pos = self.sim.data.get_site_xpos('grasp').copy()
    #     handle_pos = self.sim.data.get_site_xpos('handle_up').copy()
    #     gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
    #     drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
    #     goal_pos = goal.copy()
    #     d_eef_handle = self.goal_distance(eef_pos, handle_pos, use_xyz = True)
    #     d_handle_goal = self.goal_distance(handle_pos, goal_pos, use_xyz = False)
    #     #d_eef_goal = self.goal_distance(eef_pos, goal_pos, self.use_xyz)
    #     reward = 0
    #     reward += -2*np.sqrt(self._pos_ctrl_magnitude)
    #     reward += -5*d_eef_handle 
    #     reward_grasp = 0
    #     reward += reward_grasp + 250*(1 - 5*d_handle_goal) + 100*int(self.check_success(drawer_current_joint_pos))
    #     return reward

    #Metaworld reward
    def compute_reward(self, achieved_goal, goal, info): 
        '''
        Computes the reward of drawer close task
        : arguments : Current position achieved_goal
        : actual goal -> goal
        '''
        eef_pos = self.sim.data.get_site_xpos('grasp').copy()
        handle_pos = self.sim.data.get_site_xpos('handle_up').copy()
        drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
        goal_pos = goal.copy()
        reachDist = self.goal_distance(eef_pos, handle_pos, self.use_xyz)
        pullDist =self.goal_distance(handle_pos, goal_pos, self.use_xyz)
        reachRew = -reachDist
        reachComplete = reachDist < 0.05

        def pullReward(): 
            c1,c2,c3 = 1000,0.01,0.001
            if reachComplete: 
                pullRew = 1000*(0.3 - pullDist) + c1 * np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3)
                pullRew = max(pullRew, 0)
                return pullRew
            else: 
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew
        return reward

    def check_success(self, distance): 
        if distance < 0: 
            return True 
        else: 
            return False

    def check_contact(self,geom_1, geom_2): 
        geoms_1 = [geom_1]
        geoms_2 = [geom_2]
        for contact in self.sim.data.contact[:self.sim.data.ncon]: 
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True

            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True

            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 or c2_in_g1):  
                return True
        return False
    
    def _get_state_obs(self):
        cot_pos = self.center_of_table.copy()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        eef_pos = self.sim.data.get_site_xpos('ee_2') - cot_pos
        eef_velp = self.sim.data.get_site_xvelp('ee_2') * dt
        goal_pos = self.goal - cot_pos
        gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

        handle_pos = self.sim.data.get_site_xpos('handle') - cot_pos
        
        handle_velp = self.sim.data.get_site_xvelp('handle') * dt
        handle_velr = self.sim.data.get_site_xvelr('handle') * dt

        if not self.use_xyz:
            eef_pos = eef_pos[:2]
            eef_velp = eef_velp[:2]
            goal_pos = goal_pos[:2]
            handle_pos = handle_pos[:2]
            handle_velp = handle_velp[:2]
            handle_velr = handle_velr[:2]

        values = np.array([
            self.goal_distance(eef_pos, goal_pos, self.use_xyz),
            self.goal_distance(handle_pos, goal_pos, self.use_xyz),
            self.goal_distance(eef_pos, handle_pos, self.use_xyz),
            gripper_angle
        ])

        return np.concatenate([
            eef_pos, eef_velp, goal_pos, handle_pos, handle_velp, handle_velr, values
        ], axis=0)
    
    def _reset_sim(self):
        return BaseEnv._reset_sim(self)

    def _get_achieved_goal(self):
        return np.squeeze(self.sim.data.get_site_xpos('handle').copy())

    def quaternion_multiply(self,q,p): 
        '''
        Multiply two quaternions that represent composite rotation after one another
        q o p = [q_s * p_s - q_v^T pv, q_s * p_v + p_s*q_v + q_v X p_v]
        q = [q_s , q_v] where q_s = cos(theta/2), q_v = k*sin(theta/2) where k is axis of rotation
        p = [p_s , p_v]
        Assumption : First rotation by p then by q
        '''
        assert len(q) == 4 and len(p) == 4 , 'Quaternions must be 4 numbers'
        # assert np.linalg.norm(q) == 1 and np.linalg.norm(p) == 1, "Valid rotation quaternions must be of unit norm"
        res = np.zeros(4)
        res[0]  = q[0] * p[0] - np.dot(q[1:],p[1:])
        
        res[1:] =  (q[0]*p[1:] + p[0]*q[1:] + np.cross(q[1:], p[1:]))
        return res

    def _sample_object_pos(self):
        drawer_pos = self.sim.data.get_joint_qpos('drawer_box:joint').copy()
        drawer_quat = drawer_pos[-4:]
        drawer_pos = drawer_pos[:3]
        drawer_pos[0] += self.np_random.uniform(0.02, 0.1, size = 1)
        drawer_pos[1] += self.np_random.uniform(-0.1, 0.25, size = 1)
        k = np.asarray([0,0,1])
        theta = self.np_random.uniform(-math.pi/6, math.pi/2, size = 1)
        q_rot = np.asarray([math.cos(theta/2), k[0] * math.sin(theta/2), k[1] * math.sin(theta/2), k[2] * math.sin(theta/2)])
        final_quat = self.quaternion_multiply(q_rot, drawer_quat)
        # print(f"norm of final quaternion : {np.linalg.norm(final_quat)}")
        drawer_pose = np.concatenate([drawer_pos, final_quat])
        object_qpos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
        object_qpos += self.np_random.uniform(0.25, 0.30, size = 1)
        #object_qpos = 0.05
        self.sim.data.set_joint_qpos("drawer_box:joint", drawer_pose)
        self.sim.data.set_joint_qpos('drawer1_joint', object_qpos)

    def _sample_initial_pos(self):
        gripper_target = self.center_of_table.copy() - np.array([0.3, 0, 0])
        gripper_target[0] += self.np_random.uniform(-0.15, -0.05, size=1)
        gripper_target[1] += self.np_random.uniform(-0.05, 0.05, size=1)
        gripper_target[2] += self.default_z_offset
        if self.use_xyz:
            gripper_target[2] += self.np_random.uniform(0, 0.1, size=1)
        return BaseEnv._sample_initial_pos(self, gripper_target)
        
    def _sample_goal(self, new = False): 
        goal = self.sim.data.get_site_xpos('target01').copy()
        return BaseEnv._sample_goal(self, goal)
