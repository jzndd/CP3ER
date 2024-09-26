# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import metaworld
import random
import gymnasium as gym
# import gym
import mujoco
from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushEnvV2 as Env
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

def make_env(env_name, seed=42, render_mode=None):

    from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )
    task = f"{env_name}-goal-observable"
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
    env = env_cls(seed=seed,render_mode=render_mode)
    env._freeze_rand_vec = False
    # 相机视角设置
    env.camera_name = "corner2"
    env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
    return env

def make_env_randomized(env_name, seed=0, render_mode=None):
    Env = ALL_V2_ENVIRONMENTS[env_name] 
    env = Env(render_mode=render_mode)
    env._freeze_rand_vec = False 
    env._set_task_called = True
    return env

class MWEnvWrapper:
    def __init__(self, seed=0, env_name='push-v2', task_id=None):
        self.env_name = env_name
        self.env = make_env(env_name,seed,render_mode='rgb_array')
        self.env.model.vis.global_.offwidth = 84
        self.env.model.vis.global_.offheight = 84                  # 设置 render
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.env.mujoco_renderer.width = 84
        self.env.mujoco_renderer.height = 84
        self.env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        # set the camera id
        self.env.mujoco_renderer.camera_id = mujoco.mj_name2id(
                self.env.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "corner2",
            )
        
        # print("action dim:{},state dim:{}".format(self.act_dim, self.state_dim))

    def reset(self):
        state , _ = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminate, truncated,info = self.env.step(action.ravel())
        done = terminate or truncated
        return next_state, reward, done, info

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render()
        return frame
    
    def render_for_video(self):
        frame = self.env.render_for_video()
        return frame

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space
    
    def get_obs_space(self):
        return self.env.observation_space

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward

import numpy as np
from collections import deque
from gymnasium.spaces import Box


class ActionDTypeWrapper(gym.Wrapper):
    def __init__(self, env: MWEnvWrapper, dtype):
        self._env = env
        action_space = env.get_action_space()
        self.action_space = Box(
            low=action_space.low.astype(dtype),
            high=action_space.high.astype(dtype),
            shape=action_space.shape,
            dtype=dtype,
        )

    def step(self, action):
        action = action.astype(self.action_space.dtype)
        return self._env.step(action)

    def reset(self):
        return self._env.reset()
    
    def render(self):
        return self._env.render()
    
    def render_for_video(self):
        return self._env.render_for_video()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: ActionDTypeWrapper, num_repeats,discount):
        self._env = env
        self._num_repeats = num_repeats
        self.discount = discount

    def step(self, action):
        total_reward = 0.0
        gamma = 1.0
        for _ in range(self._num_repeats):
            observation, reward, done,info = self._env.step(action)
            total_reward += reward * gamma 
            gamma *=  self.discount
            if done:
                break

        return observation, total_reward, done, gamma , info

    def reset(self):
        return self._env.reset()
    
    def render(self):
        return self._env.render()
    
    def render_for_video(self):
        return self._env.render_for_video()

    def __getattr__(self, name):
        return getattr(self._env, name)
    

def stack_frames(frames):
    return np.concatenate(list(frames), axis=0)


class MetaWorldFrameStackWrapper(gym.Wrapper):
    def __init__(self, env: ActionRepeatWrapper, num_frames):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._env = env
        obs = env.render()
        obs_shape = obs.shape
        # remove batch dim
        if len(obs_shape) == 4:
            obs_shape = obs_shape[1:]
        new_shape = (obs_shape[2] * num_frames, obs_shape[0],obs_shape[1])
        # print("new_shape is {}".format(new_shape)) 9,480,480
        self.observation_space = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def reset(self):
        self._env.reset()
        pixels = get_pixels(self._env)
        # print("pixels shape : {}".format(pixels.shape))  3,480,480
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return stack_frames(self._frames)

    def step(self, action):
        obs, reward, done, gamma,info = self._env.step(action)
        pixels = get_pixels(self._env)
        # print("after step, pixel shape is {}".format(pixels.shape))
        self._frames.append(pixels)
        # print("after step, len(self._frames)  is {}".format(len(self._frames)))
        assert len(self._frames) == self._num_frames
        return stack_frames(self._frames), reward, done, gamma,info
    
    def render(self):
        return self._env.render()
    
    def render_for_video(self):
        return self._env.render_for_video()

def get_pixels(env: MetaWorldFrameStackWrapper):
    obs = env.render()
    # remove batch dim
    if len(obs.shape) == 4:
        obs = obs[0]
    return obs.transpose(2, 0, 1).copy()

def make(env_name, frame_stack, action_repeat, seed,discount):
    
    # env = make_env(seed=seed,render_mode='rgb_array')
    env = MWEnvWrapper(seed=seed,env_name=env_name)  # 此时 step 返回值是 state，reward,done,info
        
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, discount)  # 此时 step 返回值是 state，reward,done, gamma,info
    
    # # stack several frames
    env = MetaWorldFrameStackWrapper(env, frame_stack)

    return env
