import argparse
import numpy as np
from termcolor import colored

def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='robot', choices=['robot', 'metaworld'], type=str)
	parser.add_argument('--task_name', default='reach', type=str)
	parser.add_argument('--frame_stack', default=1, type=int)
	parser.add_argument('--observation_type', default='image', type=str, choices=["state", "image", "state+image"])
	parser.add_argument('--action_repeat', default=1, type=int)
	

	parser.add_argument('--episode_length', default=50, type=int)
	parser.add_argument('--n_substeps', default=20, type=int)
	parser.add_argument('--eval_mode', default='test', type=str)
	parser.add_argument('--action_space', default='xyzw', type=str)
	parser.add_argument('--cameras', default=0, type=int) # 0: 3rd person, 1: 1st person, 2: both
	parser.add_argument('--render', default=False, type=bool)
	
	# agent
	parser.add_argument('--algorithm', default='sacv2_3d', type=str)
	parser.add_argument('--train_steps', default='1000k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--image_size', default=84, type=int)
	parser.add_argument('--resume', default="none", type=str, help="the checkpoint path for pretrained backbone")
	parser.add_argument("--resume_rl", default="none", type=str, help="the checkpoint path for rl (include pretrained backbone)")
	parser.add_argument('--finetune', default=0, type=int, help="fine tune 3d encoder")

	
	parser.add_argument('--mean_zero', default=False, action='store_true') # normalize images to range [-0.5, 0.5] instead of [0, 1] (all)
	parser.add_argument('--predict_state', default=0, type=int)
	parser.add_argument('--hidden_dim_state', default=128, type=int)
	# --- end ---


	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=3, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=50, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# ddpg / drqv2
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--update_freq', default=2, type=int)
	parser.add_argument('--tau', default=0.01, type=float)
	parser.add_argument('--n_step', default=1, type=int)
	parser.add_argument('--num_expl_steps', default=2000, type=int)
	parser.add_argument('--std_schedule', default='linear(1.0,0.1,0.25)', type=str) # (initial, final, % of train steps)
	parser.add_argument('--std_clip', default=0.3, type=float)

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='5k', type=str)
	parser.add_argument('--eval_episodes', default=10, type=int)

	# misc
	parser.add_argument('--seed', default='1', type=str)
	parser.add_argument('--exp_suffix', default='default', type=str)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=0,choices=[0,1],type=int)
	parser.add_argument('--num_seeds', default=1, type=int)
	parser.add_argument('--visualize_configurations', default=False, action='store_true')

	#3D
	parser.add_argument('--train_rl', type=int)
	parser.add_argument('--train_3d', type=int)
	parser.add_argument('--prop_to_3d', default=1, type=int)
	parser.add_argument('--bottleneck', default=8, type=int)
	parser.add_argument('--buffer_capacity', default="-1", type=str)
	parser.add_argument('--log_3d_imgs', default="100k", type=str)
	parser.add_argument('--huber', default=0, type=int)
	parser.add_argument('--use_latent', default=0, type=int)
	parser.add_argument('--bsize_3d', default=8, type=int)
	parser.add_argument('--project_conv', default=1, type=int)
	parser.add_argument('--update_3d_freq', default=2, type=int)
	parser.add_argument('--log_train_video', default="50k", type=str)
	parser.add_argument('--augmentation', default="colorjitter", choices=["none","colorjitter","noise"], type=str) # 'colorjitter' or 'affine+colorjitter' or 'noise' or 'affine+noise' or 'conv' or 'affine+conv'
	parser.add_argument('--use_gt_camera', default=0, choices=[0,1], type=int, help="use ground truth camera param to update 3d")
	parser.add_argument("--camera_dynamic_mode", default="cycle", type=str, choices=["cycle", "line"], \
						help="use randomization mode used by dynamic camera")
	parser.add_argument("--one_more_residual_block", default=0, choices=[0,1], type=int, \
						help="add one more residual block in encoder to improve reconsturction (while hurt RL performance)")
	parser.add_argument("--camera_move_range", default=45, type=float, help="the move range of dynamic camera (degree)")
	parser.add_argument("--rand_z_axis", default=1, type=int, choices=[0,1], help="randomize the z axis of dynamic camera")

	parser.add_argument("--imagenet_normalization", default=0, choices=[0,1], type=int, help="normalize the input image with imagenet mean and std")
	parser.add_argument("--resize_to_224", default=0, choices=[0,1], type=int, help="resize the input image to 224x224") # only used for moco

	# pretrain model
	parser.add_argument("--pretrain_alg", default="video-ae", choices=["video-ae", "imagenet", "mocov3", "mocov2", "none", "vit", "r3m"])
	parser.add_argument("--pretrain_backbone", default="resnet50", choices=["resnet50", "resnet18"], type=str, \
						help="feature extraction backbone for 3d encoder when doing pretrain.")

	# finetune scale
	parser.add_argument("--lr_scale_3d_encoder", default=0.01, help="downscale the 3d learning rate", type=float)
	parser.add_argument("--lr_scale_3d_decoder", default=0.01, help="downscale the 3d learning rate",type=float)
	parser.add_argument("--lr_scale_3d_pose", default=0.01, help="downscale the 3d learning rate",type=float)
	parser.add_argument("--lr_scale_rl_backbone", default=1.0, help="downscale the backbone learning rate for rl",type=float)
	
	
	# regularization, see https://arxiv.org/pdf/1903.12436.pdf
	parser.add_argument("--regularization_3d", default=0, type=int, choices=[0,1], help="whether to perform regularization on 3d")
	parser.add_argument("--decoder_latent_lambda", default=1e-5, type=float, help="the weight of the regularization for 3d decoder")
	parser.add_argument("--decoder_weight_lambda", default=1e-7, type=float, help="the weight of the regularization for 3d decoder")
	parser.add_argument("--regularization_2d", default=0, type=int, choices=[0,1], help="whether to perform regularization on 2d")
	parser.add_argument("--weight_lambda_2d", default=1e-7, type=float, help="the weight of the regularization for 2d decoder")

	# additional loss function
	parser.add_argument('--consistency_loss_3d', default=0, type=int)
	parser.add_argument('--identity_loss_3d', default=1, type=int)
	parser.add_argument('--max_grad_norm', default=10, type=float)
	
	# dynamics
	parser.add_argument("--dynamics", default=0, type=int, help="whether to use dynamics. currently only used in sacv2_3dv2.")
	parser.add_argument("--dynamics_dims", default=512, type=int)
	parser.add_argument("--num_unroll_steps", default=3, type=int, help="unrolled steps used to update dynamics model")
	parser.add_argument("--block_size_3d", default=32, type=int, help="for computing shape of input to dynamics")

	# replay buffer
	parser.add_argument("--use_prioritized_buffer", default=0, choices=[0,1], type=int, help="whether to use prioritized replay buffer (only for 3d). default=1 because of better performance")
	parser.add_argument('--prioritized_replay_alpha', default=0.6, type=float)
	parser.add_argument('--prioritized_replay_beta', default=0.4, type=float)
	parser.add_argument('--ensemble_size', default=1, type=int)

	# wandb's setting
	parser.add_argument('--use_wandb', default=0, choices=[0,1], type=int)
	parser.add_argument('--wandb_project', default='robot_project', type=str)
	parser.add_argument('--wandb_name', default=None, type=str)
	parser.add_argument('--wandb_group', default=None, type=str)
	parser.add_argument('--wandb_job', default=None, type=str)
	parser.add_argument('--wandb_key', default=None, type=str)
	parser.add_argument('--remove_addition_log', default=0, choices=[0,1], type=int)
	parser.add_argument("--save_model_in_wandb", default=0, choices=[0,1], type=int)
	parser.add_argument("--date", type=str, default=None)


	args = parser.parse_args()

	assert args.algorithm in {'sacv2','sac', 'sacv2_3d', 'sacv2_3d_pretrain', 'sacv2_moco', 'sacv2_vit', 'sacv2_pvr', 'sacv2_mvp', 'sacv2_r3m'}, f'specified algorithm "{args.algorithm}" is not supported'
	assert (args.n_step == 1 or args.algorithm == 'drqv2') and args.n_step in {1, 3, 5}, f'n_step = {args.n_step} (default: 1) is not supported for algorithm "{args.algorithm}"'
	# assert args.image_size in {84}, f'image size = {args.image_size} (default: 84) is strongly discouraged'
	assert args.action_space in {'xy', 'xyz', 'xyzw'}, f'specified action_space "{args.action_space}" is not supported'
	assert args.eval_mode in {'train', 'test' ,'color_easy', 'color_hard', 'video_easy', 'video_hard', 'none', None}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.exp_suffix is not None, 'must provide an experiment suffix for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	
	args.train_steps = int(args.train_steps.replace('k', '000').replace('m', '000000'))

	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))
	args.buffer_capacity = int(args.buffer_capacity.replace('k', '000'))
	args.log_3d_imgs = int(args.log_3d_imgs.replace('k', '000'))
	args.log_train_video = int(args.log_train_video.replace('k', '000'))


	# select episode length. metaworld use 200. robo use 50 by default
	args.episode_length = 200 if args.domain_name == 'metaworld' else args.episode_length

	# parse seed
	args.seed = args.seed.split(',')
	if len(args.seed) == 1:
		args.seed = int(args.seed[0])
	else:
		args.seed = [int(s) for s in args.seed]

	if args.buffer_capacity == -1:
		args.buffer_capacity = args.train_steps
	
	if args.eval_mode == 'none':
		args.eval_mode = None
	
	return args
