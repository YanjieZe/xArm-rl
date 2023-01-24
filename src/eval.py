import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
#from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations
import cv2
import wandb



def evaluate(env, agent, video, num_episodes, eval_mode, image_size, test_env=True):
	episode_rewards = []
	success_rate = []

	for i in range(num_episodes):
		obs, state, info= env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs, state)
			#action = np.random.rand(2)
			obs, state, reward, done, info = env.step(action)
			video.record(env)
			episode_reward += reward
		if 'is_success' in info:
			success = float(info['is_success'])
			success_rate.append(success)

		_test_env = '_test_env'		
		video.save(f'{i}{_test_env}.mp4')

		episode_rewards.append(episode_reward)

	return np.nanmean(episode_rewards), np.nanmean(success_rate)


def main(args):

	seed_rewards = []
	for s in range(args.num_seeds):
		# Set seed
		utils.set_seed_everywhere(args.seed + s)

		# Initialize environments
		gym.logger.set_level(40)
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+s+42,
			episode_length=args.episode_length,
			action_space=args.action_space,
			n_substeps=args.n_substeps,
			frame_stack=args.frame_stack,
			image_size=args.image_size,
			cameras="dynamic", #['third_person', 'first_person']
			render=args.render, # Only render if observation type is state
			observation_type=args.observation_type,
			camera_dynamic_mode=args.camera_dynamic_mode,
			camera_move_range=args.camera_move_range,
			rand_z_axis=args.rand_z_axis,
		)


		# Set working directory
		work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, str(args.seed))
		print('Working directory:', work_dir)
		utils.make_dir(work_dir)
		assert os.path.exists(work_dir), 'specified working directory does not exist'
		model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
		video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
		video = VideoRecorder(video_dir if args.save_video else None, height=480, width=640)

		# Check if evaluation has already been run
		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
		assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

		# Prepare agent
		assert torch.cuda.is_available(), 'must have cuda enabled'
		print('Observations:', env.observation_space.shape)
		print('Actions:', env.action_space.shape)

		agent = make_agent(
			obs_shape=env.observation_space.shape,
			state_shape=env.state_space_shape,
			action_shape=env.action_space.shape,
			args=args
		)

		agent = torch.load(args.resume_rl)
		agent.train(False)

		print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
		reward, success_rate = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, args.image_size)
		print('Reward:', int(reward))
		print('Success Rate:', int(success_rate))
		seed_rewards.append(int(reward))



	print('Average Reward over all the seeds:', int(np.nanmean(seed_rewards)))


if __name__ == '__main__':
	args = parse_args()
	args.resume_rl

	# pegbox
	run_path = "rl3d/robot_pegbox/2iywwklq"
	# best_model = wandb.restore(file_path, run_path=run_path)
	file_path = "logs/robot_pegbox/sacv2_3d_pretrain/video-ae-3d1-resnet18-ft0.01-per0-L_may17real/0/model/500000.pt"
	args.resume_rl = file_path

	# # lift
	# run_path = "rl3d/robot_lift/1l25038u"
	# file_path = "logs/robot_lift/sacv2_3d_pretrain/video-ae-3d1-resnet18-ft0.01-per0-norm0-L_may26/3/model/500000.pt"
	# best_model = wandb.restore(file_path, run_path=run_path)
	# args.action_space = "xyzw"
	# args.resume_rl  = file_path
	# args.task_name = "lift"

	# # push
	# run_path = "rl3d/robot_push/3nuly118"
	# file_path = "logs/robot_push/sacv2_3d_pretrain/video-ae-3d1-resnet18-ft0.01-per0-L_may27real/4/model/500000.pt"
	# best_model = wandb.restore(file_path, run_path=run_path)
	# args.action_space = "xy"
	# args.resume_rl  = file_path
	# args.task_name = "push"

	# # reach
	# run_path = "rl3d/robot_reach/2uzlghgn"
	# file_path = "logs/robot_reach/sacv2_3d_pretrain/video-ae-3d1-resnet18-ft0.01-per0-L_may21fixed/0/model/500000.pt"
	# best_model = wandb.restore(file_path, run_path=run_path)
	# args.action_space = "xyz"
	# args.resume_rl  = file_path
	# args.task_name = "reach"

	main(args)
