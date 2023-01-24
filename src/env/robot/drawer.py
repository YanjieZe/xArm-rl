import numpy as np 
import os

from numpy.core.defchararray import join
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path

class DrawerEnv(BaseEnv, utils.EzPickle):
    def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
        self.state_dim = (22,)

        self.statefull_dim = self.state_dim
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
       
        utils.EzPickle.__init__(self)

 
   

    #Reward1
    # def compute_reward(self, achieved_goal, goal, info):
    #     eef_pos = self.sim.data.get_site_xpos('ee_2').copy()
    #     handle_pos = self.sim.data.get_site_xpos('handle').copy()
    #     gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
    #     drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
    #     goal_pos = goal.copy()
    #     d_eef_handle = self.goal_distance(eef_pos, handle_pos, self.use_xyz)
    #     d_handle_goal = self.goal_distance(handle_pos, goal_pos, use_xyz = False)
    #     reward = 0
    #     reward += -5*d_eef_handle
    #     if d_eef_handle < 0.04:
    #         reward_grasp = 15*int(self.check_contact('left_hand', 'handle_12') and self.check_contact('right_hand', 'handle_12'))
    #         if reward_grasp > 0: 
    #             reward+= reward_grasp + 250*(1 - 5*d_handle_goal) + 50*drawer_current_joint_pos
    #     return reward

    #Reward2
    # def compute_reward(self, achieved_goal, goal, info):
    #     eef_pos = self.sim.data.get_site_xpos('ee_2').copy()
    #     handle_pos = self.sim.data.get_site_xpos('handle').copy()
    #     gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
    #     drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
    #     goal_pos = goal.copy()
    #     d_eef_handle = self.goal_distance(eef_pos, handle_pos, self.use_xyz)
    #     d_handle_goal = self.goal_distance(handle_pos, goal_pos, use_xyz = False)
    #     reward = 0
    #     reward += -5*d_eef_handle
    #     if self.goal_distance(handle_pos, eef_pos, use_xyz = False) < 0.04:
    #         reward_grasp = 15*int(self.check_contact('left_hand', 'handle_12') and self.check_contact('right_hand', 'handle_12'))
    #         if reward_grasp > 0: 
    #             reward+= reward_grasp + 250*(1 - 5*d_handle_goal) + 50*drawer_current_joint_pos
    #     return reward

    # #Reward3
    # def compute_reward(self, achieved_goal, goal, info):
    #     eef_pos = self.sim.data.get_site_xpos('ee_2').copy()
    #     handle_pos = self.sim.data.get_site_xpos('handle').copy() - np.array([0, 0.03, 0.03])
    #     gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
    #     drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
    #     goal_pos = goal.copy()
    #     d_eef_handle = self.goal_distance(eef_pos, handle_pos, self.use_xyz)
    #     d_handle_goal = self.goal_distance(handle_pos, goal_pos, use_xyz = False)
    #     #d_eef_goal = self.goal_distance(eef_pos, goal_pos, self.use_xyz)
    #     reward = 0
    #     reward += 10*(1 - np.tanh(10*d_eef_handle))
    #     reward_grasp = 20*int(self.check_contact('left_hand', 'handle_12') and self.check_contact('right_hand', 'handle_12'))
    #     reward += reward_grasp
    #     if reward_grasp > 0: 
    #         reward += 250*(1 - np.tanh(10*d_handle_goal)) + 100*drawer_current_joint_pos

    #     return reward

    # #Reward4
    # def compute_reward(self, achieved_goal, goal, info):
    #     eef_pos = self.sim.data.get_site_xpos('ee_2').copy()
    #     handle_pos = self.sim.data.get_site_xpos('handle').copy() - np.array([0, 0.03, 0.03])
    #     gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
    #     drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
    #     goal_pos = goal.copy()
    #     d_eef_handle = self.goal_distance(eef_pos, handle_pos, self.use_xyz)
    #     d_handle_goal = self.goal_distance(handle_pos, goal_pos, use_xyz = False)
    #     #d_eef_goal = self.goal_distance(eef_pos, goal_pos, self.use_xyz)
    #     reward = 0
    #     reward += 10*(1 - np.tanh(10*d_eef_handle))
    #     if self.goal_distance(handle_pos, eef_pos, use_xyz = False) < 0.04:
    #         reward_grasp = 20*int(self.check_contact('left_hand', 'handle_12') and self.check_contact('right_hand', 'handle_12'))
    #         reward += reward_grasp
    #         if reward_grasp > 0: 
    #             reward += 250*(1 - np.tanh(10*d_handle_goal)) + 100*drawer_current_joint_pos

    #     return reward

    # #Reward5
    # def compute_reward(self, achieved_goal, goal, info):
    #     eef_pos = self.sim.data.get_site_xpos('grasp').copy()
    #     handle_pos = self.sim.data.get_site_xpos('handle_up').copy()
    #     gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
    #     drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
    #     goal_pos = goal.copy()
    #     d_eef_handle = self.goal_distance(eef_pos, handle_pos, self.use_xyz)
    #     d_handle_goal = self.goal_distance(handle_pos, goal_pos, use_xyz = False)
    #     #d_eef_goal = self.goal_distance(eef_pos, goal_pos, self.use_xyz)
    #     reward = 0
    #     reward += -0.25 * np.square(self._pos_ctrl_magnitude)
    #     reward += -5*d_eef_handle - 5*(gripper_angle)
    #     # if d_eef_handle < 0.04:
    #     reward_grasp = 15*int(self.check_contact('left_hand', 'handle_12') or self.check_contact('right_hand', 'handle_12'))
    #     reward += reward_grasp + 250*(1 - 5*d_handle_goal) + 100*drawer_current_joint_pos - 5*(gripper_angle)
    #     return reward

    def compute_reward_v0(self, achieved_goal, goal, info): 
        '''
        Computes the reward of drawer open task
        : arguments : Current position achieved_goal
        : actual goal -> goal
        '''
        eef_pos = self.sim.data.get_site_xpos('grasp').copy()
        handle_pos = self.sim.data.get_site_xpos('handle_up').copy()
        self.sim.data.get_site_xpos('handle')
        drawer_current_joint_pos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
        goal_pos = goal.copy()
        # first reach the up of the drawer
        reachDist = self.goal_distance(eef_pos, handle_pos, self.use_xyz)

        pullDist =self.goal_distance(handle_pos, goal_pos, self.use_xyz)
        reachRew = -reachDist
        reachComplete = reachDist < 0.05

        # 2022/02/10: add gripper open reward to enforce gripper open
        gripper_open_reward = 10 if self.current_action[-1]==1 else 0
        def pullReward(): 
            c1,c2,c3 = 1000,0.01,0.001
            if reachComplete: 
                pullRew = 1000*(0.3 - pullDist) + c1 * np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3)
                pullRew = max(pullRew, 0)
                return pullRew
            else: 
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew + gripper_open_reward
        return reward
        
    
    

    def compute_reward(self, achieved_goal, goal, info):
        return self.compute_reward_v0(achieved_goal, goal, info)
    
    def check_success(self, distance): 
        # 2022/02/05 0.02 -> 0.05
        if distance < 0.05: 
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

    def _sample_object_pos(self):

        object_qpos = self.sim.data.get_joint_qpos('drawer1_joint').copy()
        object_qpos += self.np_random.uniform(0, 0.02, size = 1)
        #object_qpos = 0.05
        self.sim.data.set_joint_qpos('drawer1_joint', object_qpos)

        self.obj_init_pos = self.sim.data.get_geom_xpos('handle_13')
        self._target_pos = self.obj_init_pos - np.array([.0, .35, .0])


    def _sample_initial_pos(self):
        drawer_pos = self.sim.data.get_body_xpos("drawer_box")
        drawer_pos[0] += np.random.uniform(-0.1, 0.1)
        drawer_pos[1] += np.random.uniform(-0.1, 0.1)


        self.sim.forward()

        gripper_target = self.center_of_table.copy() - np.array([0.3, 0, 0])
        gripper_target[0] += self.np_random.uniform(-0.15, -0.05, size=1)
        gripper_target[1] += self.np_random.uniform(-0.05, 0.05, size=1)
        gripper_target[2] += self.default_z_offset
        if self.use_xyz:
            gripper_target[2] += self.np_random.uniform(0, 0.1, size=1)
        return BaseEnv._sample_initial_pos(self, gripper_target)
        
    def _sample_goal(self, new = False): 
        goal = self.sim.data.get_site_xpos('target02').copy()
        return BaseEnv._sample_goal(self, goal)

    def _set_action(self, action):
        assert action.shape == (4,)
        
        BaseEnv._set_action(self, action)
        self.current_action = action # store current_action