import abc
import warnings

import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym
from mujoco_py import modder

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500

class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 500

    def __init__(self, model_path, frame_skip):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.np_random, _ = seeding.np_random(None)

        # used for modifying camera
        self.cam_modder = modder.CameraModder(self.sim, random_state=self.np_random)

    def seed(self, seed):
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.goal_space.seed(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render(self, offscreen=True, camera_name="corner2", resolution=(640, 480)):
        """
        where we render dynamic images for 3D view
        """

        assert_string = ("camera_name should be one of ",
                "corner3, corner, corner2, topview, gripperPOV, behindGripper, camera_static, camera_dynamic")
        assert camera_name in {"corner3", "corner", "corner2", 
            "topview", "gripperPOV", "behindGripper", "camera_static", "camera_dynamic"}, assert_string
        if not offscreen:
            self._get_viewer('human').render()
        else:
            # We manually change the camera mode in `xyz_base.xml`
            # When using target mode of camera, the image is fliped vertically, so we need to flip back.
            USE_TARGET_MODE = True

            img_obs = self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name
            )

            if USE_TARGET_MODE: # flip the image vertically if using target mode of camera
                img_obs = np.flipud(img_obs)
            return img_obs

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
            
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
        
    def _get_state_obs(self):
        raise NotImplementedError('_get_state_obs has not been implemented for this task!')

    def _get_achieved_goal(self):
        raise NotImplementedError('_get_achieved_goal has not been implemented for this task!')


    
    def get_robot_state_obs(self):
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)

        return np.concatenate([pos_hand, np.array([gripper_distance_apart])]).astype(np.float32)

    def _get_image_obs(self, camera_name="camera_static", resolution=(128,128)):
        return self.render(offscreen=True, camera_name=camera_name, resolution=resolution)
    
    def render_obs(self, mode=None, width=448, height=448, camera_id=None):
        self.sim.forward()
        cameras = ['camera_static', 'camera_dynamic']
        data = list()
        for camera_name in cameras:
            data.append(
                self.render(offscreen=True, camera_name=camera_name, resolution=(width, height))
            )
        return np.stack(data)

    
