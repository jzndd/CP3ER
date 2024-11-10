import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import mw

import utils
from logger import Logger
from wandblogger import WandbLogger
from replay_buffer_mw import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from datetime import datetime

from omegaconf import OmegaConf

from gymnasium.spaces import Box

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_space,
                                self.train_env.action_space,
                                self.cfg.agent)
        print("-----------------------------------make agent successfully----------------------------------------------------")
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # # create wandb logger
        if self.cfg.use_wb:
            print("-----------------------------------load wandb----------------------------------------------------")
            wandb_config = OmegaConf.to_container(self.cfg, resolve=True)
            self.wandblogger = WandbLogger(self.cfg.task_name,"cp3er",config=wandb_config, seed=self.cfg.seed)
        
        # create envs
        self.train_env = mw.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, self.cfg.discount)
        self.eval_env = mw.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, self.cfg.discount)
        
        
        # create replay buffer
        data_specs = (self.train_env.observation_space,  # obs
                      self.train_env.action_space,        # act
                      Box(low=-np.inf, high=np.inf,shape=(1,), dtype=np.float32), # reward
                      Box(low=0, high=1,shape=(1,), dtype=np.float32), # discount
        )

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, sample_alpha=self.cfg.sample_alpha)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def get_action(self, obs):
        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs)
        return action

    def eval(self):
        # All steps are agent steps (including action repeat)
        step = 0
        num_episodes = 0
        total_reward = 0
        total_success = 0
        total_first_success_step = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(num_episodes):
            obs = self.eval_env.reset()
            done = False
            success = False
            current_steps = 0
            first_success_step = 0
            self.video_recorder.init(self.eval_env, enabled=(num_episodes == 0))
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs)
                    # action = self.agent.act(time_step.observation)

                obs, reward, done, discount,info = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += reward
                success |= bool(info['success'])

                step += 1
                current_steps += 1

                if first_success_step == 0 and info['success']:  # just record the first time success
                    first_success_step = current_steps
                    print("{}:agent get success in {}, and done is {} , return is {}".format(current_steps,first_success_step,done,total_reward))

            if success:
                total_success += 1
                total_first_success_step += first_success_step

            num_episodes += 1

            if self.global_frame > 1900000:
                self.video_recorder.save(f'{self.global_frame}.mp4')

        avg_episode_reward = total_reward / num_episodes
        success_rate = total_success / num_episodes
        avg_first_success_step = total_first_success_step / total_success if total_success > 0 else 0.

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', avg_episode_reward)
            log('episode_length', step * self.cfg.action_repeat / num_episodes)
            log('episode', self.global_episode)
            log('step', self.global_step)

        if self.cfg.use_wb:    
            self.wandblogger.scalar_summary("eval/episode_reward",avg_episode_reward, self.global_frame)
            self.wandblogger.scalar_summary('eval/episode_length', step * self.cfg.action_repeat / num_episodes, self.global_frame)
            self.wandblogger.scalar_summary("eval/episode",self.global_episode, self.global_frame)
            self.wandblogger.scalar_summary("eval/success_rate",success_rate, self.global_frame)
            self.wandblogger.scalar_summary("eval/first_success_step",avg_first_success_step, self.global_frame)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)

        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        obs = self.train_env.reset()
        # self.replay_storage.add(time_step)
        self.train_video_recorder.init(obs)
        metrics = None
        done = False
        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                    if self.cfg.use_wb:

                        self.wandblogger.scalar_summary("train/fps", episode_frame / elapsed_time, self.global_frame)
                        self.wandblogger.scalar_summary('train/total_time', total_time, self.global_frame)
                        self.wandblogger.scalar_summary('train/episode_reward', episode_reward, self.global_frame)
                        self.wandblogger.scalar_summary('train/episode_length', episode_frame, self.global_frame)
                        self.wandblogger.scalar_summary('train/episode', self.global_episode, self.global_frame)
                        self.wandblogger.scalar_summary('train/buffer_size', len(self.replay_storage), self.global_frame)
                        self.wandblogger.scalar_summary('train/step', self.global_step, self.global_frame)

                # reset env
                obs = self.train_env.reset()
                # self.replay_storage.add(time_step)
                self.train_video_recorder.init(obs)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                if self.cfg.save_checkpoint and self.global_frame % 500000 == 0:
                    self.save_ckpt()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.use_wb:
                    self.wandblogger.scalar_summary("eval/eval_total_time", self.timer.total_time(), self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.global_frame < self.agent.num_expl_steps: #  default is 10k
                    action = self.train_env._env._env._env.get_random_action()
                else:
                    action = self.agent.act(obs)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if self.cfg.use_wb:
                    self.wandblogger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            obs, reward, done, discount,info = self.train_env.step(action)
            episode_reward += reward
            self.replay_storage.add(obs, action, reward, done,discount)
            self.train_video_recorder.record(obs)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'ckpt1500000.pt'
        with snapshot.open('rb+') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def save_ckpt(self):
        snapshot = self.work_dir / 'ckpt{}.pt'.format(self.global_frame)
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train_mw import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
