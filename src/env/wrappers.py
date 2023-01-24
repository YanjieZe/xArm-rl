import numpy as np
from numpy.random import randint
import os
import gym
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from gym.wrappers import TimeLimit
from env.robot.registration import register_robot_envs
from env.metaworld import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import utils
from collections import deque
from mujoco_py import modder
import copy
import math
from algorithms import rot_utils
from termcolor import colored

def make_env(
        domain_name,
        task_name,
        seed=0,
        episode_length=50,
        n_substeps=20,
        frame_stack=1,
        image_size=84,
        cameras="dynamic",
        render=False,
        observation_type='image',
        action_space='xyzw',
        camera_dynamic_mode="cycle",
        camera_move_range=30,
        rand_z_axis=0,
        action_repeat=1,
):
    """Make environment for experiments"""
    # assert domain_name == 'robot', f'expected domain_name "robot", received "{domain_name}"'
    assert action_space in {'xy', 'xyz', 'xyzw'}, f'unexpected action space "{action_space}"'

    print("TYPE ", observation_type)
    register_robot_envs(
        n_substeps=n_substeps,
        observation_type=observation_type,
        image_size=image_size,
        use_xyz=action_space.replace('w', '') == 'xyz')
    

    randomizations = {'camera'}
    

    assert cameras=='dynamic', "Please specify cameras as dynamic."

    if domain_name == 'robot':
        env_id = 'Robot' + task_name.capitalize() + '-v0'
        env = gym.make(env_id, cameras=cameras, render=render, observation_type=observation_type)
        
        env.seed(seed)
        env.task_name = task_name
        env = TimeLimit(env, max_episode_steps=episode_length)
        env = SuccessWrapper(env, any_success=True)
        env = ObservationSpaceWrapper(env, observation_type=observation_type, image_size=image_size)
        env = ActionSpaceWrapper(env, action_space=action_space, action_repeat=action_repeat)
        env = FrameStack(env, frame_stack)
        env = DynamicCameraWrapper(env, domain_name=domain_name, camera_move_range=camera_move_range, seed=seed)
        # env = DomainRandomizationWrapper(env, domain_name=domain_name, randomizations=randomizations, seed=seed,
        #                                 camera_name=cameras, camera_dynamic_mode=camera_dynamic_mode, camera_move_range=camera_move_range, 
        #                                 rand_z_axis=rand_z_axis)
        env = CameraPosWrapper(env)
    elif domain_name == "metaworld":
        env_id = task_name + '-v2-goal-hidden'
        env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]()
        env.seed(seed)
        env.task_name = task_name
        env = TimeLimit(env, max_episode_steps=episode_length)
        env = MetaworldSuccessWrapper(env, any_success=True)
        env = MetaworldObservationSpaceWrapper(env, observation_type=observation_type, image_size=image_size)
        env = ActionSpaceWrapper(env, action_space=action_space)
        env = FrameStack(env, frame_stack)
        env = DynamicCameraWrapper(env, domain_name=domain_name, camera_move_range=camera_move_range, seed=seed)
        env = CameraPosWrapper(env)

        

        


    return env


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        if len(shp) == 3:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(shp[0] * k,),
                dtype=env.observation_space.dtype
            )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, state_obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), state_obs

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), state,  reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


class SuccessWrapper(gym.Wrapper):
    def __init__(self, env, any_success=True):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.any_success = any_success
        self.success = False

    def reset(self):
        self.success = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.any_success:
            self.success = self.success or bool(info['is_success'])
        else:
            self.success = bool(info['is_success'])
        info['is_success'] = self.success
        return obs, reward, done, info


class MetaworldSuccessWrapper(gym.Wrapper):
    def __init__(self, env, any_success=True):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.any_success = any_success
        self.success = False

    def reset(self):
        self.success = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.any_success:
            self.success = self.success or bool(info['success'])
        else:
            self.success = bool(info['success'])
        info['is_success'] = info['success'] = self.success
        return obs, reward, done, info

class ObservationSpaceWrapper(gym.Wrapper):
    def __init__(self, env, observation_type, image_size):
        # assert observation_type in {'state', 'image'}, 'observation type must be one of \{state, image\}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size


       

        if self.observation_type in ['image', 'state+image']:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3 * 2, image_size, image_size),
                                                    dtype=np.uint8)

        elif self.observation_type == 'state':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim,
                                                    dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs), obs['state'] if 'state' in obs else None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), obs['state'] if 'state' in obs else None, reward, done, info

    def _get_obs(self, obs_dict):
        obs = obs_dict['observation']
        if self.observation_type in ['image', "state+image"]:
            output = np.empty((3 * obs.shape[0], self.image_size, self.image_size), dtype=obs.dtype)
            for i in range(obs.shape[0]):
                output[3 * i: 3 * (i + 1)] = obs[i].transpose(2, 0, 1)
        elif self.observation_type == 'state':
            output = obs_dict['observation']
        return output

class MetaworldObservationSpaceWrapper(gym.Wrapper):
    """
    Provide more modalities for metaworld env.
    
    Originally, metaworld gives state by default. Our wrapper gives robot state (gripper distance and position) and image obs as well.
    """

    def __init__(self, env, observation_type, image_size):
        assert observation_type in {'state', 'image', 'state+image'}, 'observation type must be one of \{state, image, state+image\}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size

        self.state_space_shape = (4,) # the state refers to robot state
        if self.observation_type in {'state', 'image'}:
            self.state_space_shape = None


        if self.observation_type in ['image', 'state+image']:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3 * 2, image_size, image_size),
                                                    dtype=np.uint8)
        elif self.observation_type == 'state':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim,
                                                    dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        obs_dict = self._get_obs(obs)
        return obs_dict['observation'], obs_dict['robot_state']

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # obs is `state` now. and we fetch other modalities by `_get_obs`
        obs_dict = self._get_obs(obs)
        return obs_dict['observation'], obs_dict['robot_state'], reward, done, info


    def _get_obs(self, obs):
        obs_dict = dict()
        # fetch img obs or state obs
        if self.observation_type in ['image', "state+image"]:
            img_obs_origin = self.render_obs(width=self.image_size, height=self.image_size)
            img_obs = np.empty((3 * 2, self.image_size, self.image_size), dtype=obs.dtype)
            img_obs[:3] = img_obs_origin[0].transpose(2, 0, 1)
            img_obs[3:] = img_obs_origin[1].transpose(2, 0, 1)
            obs_dict['observation'] = img_obs
        elif self.observation_type == 'state':
            obs_dict['observation'] = obs

        # fetch robot state
        if self.observation_type in ['state+image']:
            obs_dict['robot_state'] = self.get_robot_state_obs()
        else:
            obs_dict['robot_state'] = None

        return obs_dict


class ActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, action_space, action_repeat=1):
        assert action_space in {'xy', 'xyz', 'xyzw'}, 'task must be one of {xy, xyz, xyzw}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.action_space_dims = action_space
        self.use_xyz = 'xyz' in action_space
        self.use_gripper = 'w' in action_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 + self.use_xyz + self.use_gripper,),
                                           dtype=np.float32)
        self.action_repeat = action_repeat

    def step(self, action):
        assert action.shape == self.action_space.shape, 'action shape must match action space'
        action = np.array(
            [action[0], action[1], action[2] if self.use_xyz else 0, action[3] if self.use_gripper else 1],
            dtype=np.float32)
        for i in range(self.action_repeat - 1):
            _ = self.env.step(action)
        return self.env.step(action)



class CameraPosWrapper(gym.Wrapper):
    """
    Record camera pos and return
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        obs, state = self.env.reset()
        
        info = {}
        info["camera_RT"] = self.get_camera_RT()
        info["camera_intrinsic"] = self.get_camera_intrinsic()
        info["camera_extrinsic"] = self.get_camera_extrinsic()
        info["focal_length"] = self.get_focal_length()

        return obs, state, info

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
    
        info["camera_RT"] = self.get_camera_RT()
        info["camera_intrinsic"] = self.get_camera_intrinsic()
        info["camera_extrinsic"] = self.get_camera_extrinsic()
        info["focal_length"] = self.get_focal_length()

        return obs, state, reward, done, info


    def get_camera_RT(self):
        """
        get camera front/dynamic 's [eluer angle, translation]

        eluer angle order: roll, pitch, yaw
        """
        camera_pos_front = self.cam_modder.get_pos("camera_static")
        camera_quat_front = self.cam_modder.get_quat("camera_static")
        camera_eluer_front = self.euler_from_quaternion(camera_quat_front)
        camera_param_front = np.hstack([camera_eluer_front, camera_pos_front])

        camera_pos_dynamic = self.cam_modder.get_pos("camera_dynamic")
        camera_quat_dynamic = self.cam_modder.get_quat("camera_dynamic")
        camera_eluer_dynamic = self.euler_from_quaternion(camera_quat_dynamic)
        camera_param_dynamic = np.hstack([camera_eluer_dynamic, camera_pos_dynamic])

        camera_param = np.hstack([camera_param_front, camera_param_dynamic])
        # print("front eluer", camera_eluer_front, "dynamic eluer", camera_eluer_dynamic)
        return camera_param

    def get_camera_extrinsic(self):
        """
        get camera extrinsic, 3x4 matrix
        """
        camera_pos_front = self.cam_modder.get_pos("camera_static")
        camera_quat_front = self.cam_modder.get_quat("camera_static")
        rotation_matrix_front = self.quat2mat(camera_quat_front)
        
        # get extrinsic matrix
        camera_extrinsic_front = np.eye(4)
        camera_extrinsic_front[:3, :3] = rotation_matrix_front
        camera_extrinsic_front[:3, 3] = camera_pos_front


        camera_pos_dynamic = self.cam_modder.get_pos("camera_dynamic")
        camera_quat_dynamic = self.cam_modder.get_quat("camera_dynamic")
        rotation_matrix_dynamic = self.quat2mat(camera_quat_dynamic)

        # get extrinsic matrix
        camera_extrinsic_dynamic = np.eye(4)
        camera_extrinsic_dynamic[:3, :3] = rotation_matrix_dynamic
        camera_extrinsic_dynamic[:3, 3] = camera_pos_dynamic

        # stack [static, dynamic], get 2x4x4 matrix
        camera_extrinsics = np.stack([camera_extrinsic_front, camera_extrinsic_dynamic], axis=0)

        return camera_extrinsics
      

    def get_camera_intrinsic(self):

        img_height = img_width = self.image_size

        fovys = self.sim.model.cam_fovy
        assert fovys[0]==fovys[1], "two cameras should use same fovy"
        fovy = fovys[0]

        f = 0.5 * img_height / math.tan(fovy * math.pi / 360)

        intrinsic = np.array(((f, 0, img_width / 2), (0, f, img_height / 2), (0, 0, 1)))

        return intrinsic

    def get_focal_length(self):
        """
        Focal length (f) and field of view (FOV) of a lens are inversely proportional. 
        For a standard rectilinear lens, FOV = 2 arctan x/2f
        """
        img_height = img_width = self.image_size
        fovys = self.sim.model.cam_fovy
        assert fovys[0]==fovys[1], "two cameras should use same fovy"
        fovy = fovys[0]
        focal_length = 0.5 * img_height / math.tan(fovy * math.pi / 360)
        return focal_length

    def euler_from_quaternion(self, quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def quat2mat(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
        
        return rot_matrix
                            


class DynamicCameraWrapper(gym.Wrapper):
    """
    wrapper for randomizing camera
    """
    def __init__(self, env, domain_name, camera_move_range=30, seed=None):
        gym.Wrapper.__init__(self, env)
        self.random_state = np.random.RandomState(seed)
        self.cam_modder = modder.CameraModder(self.sim, random_state=self.random_state)

        self.camera_move_range = np.deg2rad(camera_move_range)

        if domain_name == "robot":
            # for fixed mode
            self.start_angle = np.deg2rad(-30) # depends on env
            self.interpolation_step = 0.02 # depends on episode length, 1/50 = 0.02
            self.camera_rotation_radius = 0.85 # depends on env
            
        elif domain_name == "metaworld":

            # for fixed mode
            # self.interpolation_step = 0.005
            # self.start_angle = np.deg2rad(120)
            # self.camera_rotation_radius = 1.3 # for fixed mode

            # for target mode
            self.interpolation_step = 0.005 # 1/200 = 0.005
            self.start_angle = np.deg2rad(-90) 
            self.camera_rotation_radius = 0.55 # for target mode
        else:
            raise NotImplementedError("domain {} is not implemented".format(domain_name))


        self.record_inital_camera_pos()
        self.compute_camera_rotation_base()
        self.compute_interpolate_trajectory()
    

    def step(self, action):
        self._randomize_camera()
        obs, state, reward, done, info = self.env.step(action)
        return obs, state, reward, done, info
    
    def reset(self):
        return self.env.reset()

    def record_inital_camera_pos(self):
        """
        Record new initialized camera.
        """
        self.init_camera_positions = {}
        self.init_camera_positions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_pos("camera_dynamic"))
        self.init_camera_positions["camera_static"] = copy.deepcopy(self.cam_modder.get_pos("camera_static"))
        
        self.init_camera_quaternions = {}
        self.init_camera_quaternions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_quat("camera_dynamic"))
        self.init_camera_quaternions["camera_static"] = copy.deepcopy(self.cam_modder.get_quat("camera_static"))


    def compute_interpolate_trajectory(self):
        """
        Use front and dynamic view to generate interpolated traj
        """
        
        self.first_pos = copy.deepcopy(self.init_camera_positions["camera_static"])
        self.second_pos = copy.deepcopy(self.init_camera_positions["camera_dynamic"])
        self.traj_idx = 0 
        
        self.camera_traj = []
        self.roll_traj = []

       
        interpolation_sequence = np.arange(0.0, 0.99, self.interpolation_step)
        self.traj_len = interpolation_sequence.shape[0]
        print(colored("camera traj len: %u"%self.traj_len, color="cyan"))

        for a in interpolation_sequence:
            self.camera_traj.append( self.start_angle + a*self.camera_move_range ) 
            self.roll_traj.append( self.base_roll + a*self.camera_move_range )



    def compute_camera_rotation_base(self):
        # use the static camera to compute the centre, and apply on dynamic camera
        pos = self.cam_modder.get_pos("camera_static")
        self.base_x = pos[0] -  self.camera_rotation_radius * np.sin(self.start_angle)
        self.base_y = pos[1] -  self.camera_rotation_radius * np.cos(self.start_angle)
        self.base_z = pos[2]

        # and get base euler
        quat = self.cam_modder.get_quat("camera_static")
        self.base_roll, self.base_pitch, self.base_yaw = rot_utils.quat2euler(*quat)



    def _randomize_camera(self, rand_first=False):
        """
        The core of the dynamic camera moving.
        """

        # set dynamic camera, which is moving
        theta = self.camera_traj[self.traj_idx]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)


        # change translation
        pos = self.cam_modder.get_pos("camera_dynamic")

        pos[0] = self.base_x + self.camera_rotation_radius * sin_theta
        pos[1] = self.base_y + self.camera_rotation_radius * cos_theta
        pos[2] = self.base_z
        
        self.cam_modder.set_pos("camera_dynamic", pos)


        # change rotation
        quat = self.cam_modder.get_quat("camera_dynamic")
        roll, pitch, yaw = rot_utils.quat2euler(*quat)
        roll = self.roll_traj[self.traj_idx]
        pitch = self.base_pitch
        yaw = self.base_yaw
        quat = rot_utils.euler2quat(roll, pitch, yaw)
        self.cam_modder.set_quat("camera_dynamic", quat)

        # increment traj idx
        self.traj_idx = (self.traj_idx + 1)%self.traj_len


    def reset(self):
        self.traj_idx = 0 # reset dynamic camera
        obs, state= self.env.reset()
        return obs, state
    
    def change_traj_idx(self, idx):
        self.traj_idx = idx % self.traj_len
        



class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, domain_name, camera_name, camera_dynamic_mode="cycle", camera_move_range=45, rand_z_axis=1, randomizations=None, seed=None):

        # assert domain_name in {'reach', 'push', 'cloth'}, \
        #	'domain randomization only implemented for reach, push, cloth domains'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.domain_name = domain_name
        self.valid_randomizations = {'camera', 'light', 'material', 'skybox', 'texture', 'brightness'}
        if randomizations is None:
            randomizations = {}
        assert isinstance(randomizations, (dict, set)), \
            'randomizations must be one of {dict, set, "all", None}'
        for randomization in randomizations:
            assert randomization in self.valid_randomizations, \
                f'received invalid randomization: "{randomization}"'
        self.randomizations = randomizations
        self.sim = self.env.unwrapped.sim

        self.camera_dynamic_mode = camera_dynamic_mode
        self.camera_move_range = np.deg2rad(camera_move_range)
        self.rand_z_axis = rand_z_axis

        self.random_state = np.random.RandomState(seed)
        self.camera_name = "camera_" + camera_name
        self.cam_modder = modder.CameraModder(self.sim, random_state=self.random_state)
        self.light_name = 'light0'
        self.light_modder = modder.LightModder(self.sim, random_state=self.random_state)
        self.geom_names = ['tablegeom0', 'floorgeom0']
        self.material_modder = modder.MaterialModder(self.sim, random_state=self.random_state)
        self.texture_modder = modder.TextureModder(self.sim, random_state=self.random_state)
        self.brightness_std = 0


        self.record_inital_camera_pos() # record camera pos
        self.compute_interpolate_trajectory()

    def reset(self):
        self.traj_idx = 0
        if 'texture' in self.randomizations:
            self._randomize_texture()
        if 'camera' in self.randomizations:
            self._randomize_camera(rand_first=False)
        if 'light' in self.randomizations:
            self._randomize_light()
        if 'material' in self.randomizations:
            self._randomize_material()
        if 'skybox' in self.randomizations and not 'texture' in self.randomizations:
            self._randomize_skybox()
        if 'brightness' in self.randomizations:
            self._randomize_brightness()
        obs, state= self.env.reset()
        return self._modify_obs(obs), state
    
    def change_traj_idx(self, idx):
        self.traj_idx = idx % self.traj_len
        
    def step(self, action):
        self._randomize_camera()
        obs, state, reward, done, info = self.env.step(action)
        return self._modify_obs(obs), state, reward, done, info

    def _modify_obs(self, obs):
        """if len(obs.shape) > 1:
			return (np.clip(obs/255. + np.ones_like(obs) * self.brightness_std, 0, 1)*255).astype(np.uint8)"""
        return obs

    def _randomize_texture(self):
        for name in self.geom_names:
            self.texture_modder.whiten_materials()
            self.texture_modder.set_checker(name, (255, 0, 0), (0, 0, 0))
            self.texture_modder.rand_all(name)
        self.texture_modder.rand_gradient('skybox')


    def get_new_position(self, pos, camera_name):

        if camera_name not in ['camera_static', 'camera_dynamic']:
            raise Exception('Wrong Camera Name in Randomness Wrapper.')
        init_position = copy.deepcopy(self.init_camera_positions[camera_name])
        new_x = init_position [0]
        new_x += 0.3*(np.random.rand()-0.5)

        new_y = init_position [1]
        new_y += 0.3*(np.random.rand()-0.5)

        new_z = init_position [2] 
        new_z += 0.1*(np.random.rand()-0.5)
  
        return np.array([new_x,new_y,new_z])


    def _randomize_camera(self, rand_first=False):
        # if self.task_name == 'shelfplacing':
        #     self._randomize_camera_cycle_for_shelfplacing(rand_first=rand_first)
        #     return
            
        if self.camera_dynamic_mode == "cycle":
            self._randomize_camera_cycle(rand_first=rand_first)
        else:
            self._randomize_camera_line(rand_first=rand_first)

    def _randomize_camera_line(self, rand_first=False):

        # fetch one pos in traj
        pos = self.camera_traj[self.traj_idx]
        
        self.traj_idx = (self.traj_idx + 1)%self.traj_len

        # some randomness
        pos = self.get_new_position(pos, "camera_dynamic")

        self.cam_modder.set_pos("camera_dynamic", pos)

        if rand_first:
            pos = self.cam_modder.get_pos("camera_static")
          
            pos = self.get_new_position(pos, "camera_static")

            self.cam_modder.set_pos("camera_static", pos)

    def record_inital_camera_pos(self):
        """
        Record new initialized camera.
        """
        self.init_camera_positions = {}
        self.init_camera_positions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_pos("camera_dynamic"))
        self.init_camera_positions["camera_static"] = copy.deepcopy(self.cam_modder.get_pos("camera_static"))
        

    def compute_interpolate_trajectory(self):
        """
        Use static and dynamic view to generate interpolated traj
        """
        
        self.first_pos = copy.deepcopy(self.init_camera_positions["camera_static"])
        self.second_pos = copy.deepcopy(self.init_camera_positions["camera_dynamic"])
        self.traj_idx = 0 
        
        self.camera_traj = []
        if self.camera_dynamic_mode == "line":
            interpolation_sequence = np.arange(0.0, 1.1, 0.1)
            self.traj_len = interpolation_sequence.shape[0]
            for a in interpolation_sequence:
                self.camera_traj.append( (1-a)*self.first_pos + a*self.second_pos ) 
        elif self.camera_dynamic_mode == "cycle":
            interpolation_step = 0.02
            interpolation_sequence = np.arange(0.0, 0.99, interpolation_step)
            rotation_angle = self.camera_move_range
            self.start_angle = np.deg2rad(-25)
            self.traj_len = interpolation_sequence.shape[0]
            print(colored("camera traj len: %u"%self.traj_len, color="cyan"))

            for a in interpolation_sequence:
                self.camera_traj.append( self.start_angle + a*rotation_angle ) 



    def _randomize_camera_cycle(self, rand_first=False):
        """
        The core of the dynamic camera moving.
        """

        base_x, base_y, base_z = 1.3, 0.4, 1.2
        x_radius = 0.6
        y_radius = 0.6
        # set front camera, which is fixed
        pos = self.cam_modder.get_pos("camera_dynamic")
        pos[0] = base_x + x_radius * np.sin(self.start_angle)
        pos[1] = base_y + y_radius * np.cos(self.start_angle)
        pos[2] = base_z
        self.cam_modder.set_pos("camera_dynamic", pos)


        # set dynamic camera, which is moving
        theta = self.camera_traj[self.traj_idx]
        self.traj_idx = (self.traj_idx + 1)%self.traj_len

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        pos = self.cam_modder.get_pos("camera_dynamic")

        if self.rand_z_axis:
            z_noise = 0.2*(np.random.random())
        else:
            z_noise = 0

        new_x = base_x + x_radius * sin_theta
        new_y = base_y + y_radius * cos_theta
        new_z = base_z + z_noise
        

        pos[0] = new_x
        pos[1] = new_y
        pos[2] = new_z

        self.cam_modder.set_pos("camera_dynamic", pos)



    def _randomize_brightness(self):
        self.brightness_std = self._add_noise(0, 0.05)

    def _randomize_light(self):
        self.light_modder.set_ambient(self.light_name, self._uniform([0.2, 0.2, 0.2]))
        self.light_modder.set_diffuse(self.light_name, self._uniform([0.8, 0.8, 0.8]))
        self.light_modder.set_specular(self.light_name, self._uniform([0.3, 0.3, 0.3]))
        self.light_modder.set_pos(self.light_name, self._add_noise([0, 0, 4], 0.5))
        self.light_modder.set_dir(self.light_name, self._add_noise([0, 0, -1], 0.25))
        self.light_modder.set_castshadow(self.light_name, self.random_state.randint(0, 2))

    def _randomize_material(self):
        for name in self.geom_names:
            self.material_modder.rand_all(name)

    def _randomize_skybox(self):
        self.texture_modder.rand_gradient('skybox')
        geom_id = self.sim.model.geom_name2id('floorgeom0')
        mat_id = self.sim.model.geom_matid[geom_id]
        self.sim.model.mat_rgba[mat_id] = np.clip(self._add_noise([0.2, 0.15, 0.1, 1], [0.1, 0.2, 0.2, 0]), 0, 1)

    def _uniform(self, default, low=0.0, high=1.0):
        if isinstance(default, list):
            default = np.array(default)
        if isinstance(low, list):
            assert len(low) == len(default), 'low and default must be same length'
            low = np.array(low)
        if isinstance(high, list):
            assert len(high) == len(default), 'high and default must be same length'
            high = np.array(high)
        return np.random.uniform(low=low, high=high, size=len(default))

    def _add_noise(self, default, std):
        if isinstance(default, list):
            default = np.array(default)
        elif isinstance(default, (float, int)):
            default = np.array([default], dtype=np.float32)
        if isinstance(std, list):
            assert len(std) == len(default), 'std and default must be same length'
            std = np.array(std)
        return default + std * self.random_state.randn(len(default))
