import torch
import os

import numpy as np
import gym
import utils
import time
import wandb
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import tqdm
from PIL import Image
from torchvision.utils import make_grid, save_image

def cover_img1_on_img2(path1, path2):
	
	img1 = Image.open(path1)
	img2 = Image.open(path2) # 640 * 480
	# left, top, right, bottom
	img2 = img2.crop((146, 57, 515, 430))
	img2 = img2.resize((260,260))
	img2.putalpha(60)
	# img2.save("sim2real_imgs/real_alpha.png")
	img1.paste(img2, (0,0), img2)
	img1.save("sim2real_imgs/cover.png")

def visualize_configurations(env, args):

	
	frames = []
	figure_num=6
	env.reset()
	for i in tqdm.tqdm(range(figure_num)):
		# env.seed(i)
		# env.reset()
		# random step
		for _ in range(10):
			action = env.action_space.sample()
			obs, state, reward, done, info = env.step(action)
		frame = torch.from_numpy(env.render_obs(mode='rgb_array',
				height=args.image_size, width=args.image_size, camera_id=None).copy()).squeeze(0)
		frame_to_store = torch.from_numpy(obs[3:]).div(255).unsqueeze(0)
		frame0 = frame[0].permute(2,0,1).float().div(255)
		frame1 = frame[1].permute(2,0,1).float().div(255)
		frames.append(frame0)
		frames.append(frame1)
		if not os.path.exists("env_imgs"):
			os.mkdir("env_imgs")
		if not os.path.exists("env_imgs_paper"):
				os.mkdir("env_imgs_paper")
		save_image(frame_to_store, f'env_imgs_paper/{args.domain_name}_{args.task_name}_{str(args.image_size)}_{str(i)}.png')

	save_image(make_grid(torch.stack(frames), nrow=1), f'env_imgs/{args.domain_name}_{args.task_name}_{str(args.image_size)}.png')
		



def videolize_configurations(env, args, camera="front"):
	video_dir = "env_videos"
	fps=20
	if not os.path.exists(video_dir):
		os.mkdir(video_dir)

	video = VideoRecorder(video_dir if args.save_video else None, height=args.image_size, width=args.image_size, fps=fps)
	episode_rewards = []
	success_rate = []

	obs = env.reset()
	video.init(enabled=1)
	done = False
	episode_reward = 0
	step=0
	debug = False

	while not done:
		action = env.action_space.sample()
		obs, state, reward, done, info = env.step(action)

		if debug:
			test_img = torch.from_numpy(obs[:]).float().div(255)
			save_image( test_img[3:], "debug.png" )
			print("second pos:", info["camera_RT"][6:])

		video.record(env, camera=camera, domain_name=args.domain_name)
		print("step: %u | reward: %f"%(step, reward))
		step += 1
		episode_reward += reward

	print("is success:", info['is_success'] )
	if camera is None:
		camera="all"
	rand_z_axis = args.rand_z_axis
    	
	video.save(f'{args.domain_name}_{args.task_name}_{str(args.image_size)}_{str(args.camera_move_range)}.mp4')
	print("video saved. episode reward is %f"%episode_reward)

def visualize_env(args):
	cameras = "static"
	cameras = "dynamic"
	cameras = None

	utils.set_seed_everywhere(args.seed)

	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		cameras="dynamic", #['front', 'dynamic']
		render=args.render, # Only render if observation type is state
		observation_type="state+image", # state, image, state+image
		action_space=args.action_space,
		camera_dynamic_mode=args.camera_dynamic_mode,
		camera_move_range=args.camera_move_range,
		rand_z_axis=args.rand_z_axis,
		action_repeat=args.action_repeat,
	)
	

	visualize_configurations(env=env, 
							args=args)

	if args.save_video:
		videolize_configurations(env=env, 
								args=args,
								camera=cameras)

if __name__=='__main__':
	args = parse_args()
	visualize_env(args)

	path2 = "sim2real_imgs/shelf_real.png"
	# path2 = "sim2real_imgs/real_alpha.png"
	path1 =  "env_imgs/robot_shelfreal_256.png"
	# cover_img1_on_img2(path1, path2)

	
