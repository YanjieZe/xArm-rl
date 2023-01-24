import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import json
import utils
sns.set(style='whitegrid', rc={'xtick.direction': 'inout', 'ytick.direction': 'inout', 'grid.linestyle': ':'})

color_palette = [
	'tab:blue',
	'tab:orange',
	'black',
	'tab:green',
	'tab:purple',
	'tab:red',
	'tab:cyan',
	'tab:brown',
	'tab:olive',
	'xkcd:gold',
	'xkcd:mint'
]


def running_mean(x, N):
	x = np.pad(x, (N//2, N-1-N//2), mode='reflect')
	x = np.convolve(x, np.ones((N,))/N, mode='valid')
	return x


def log_path_to_numpy(fp, train):
	with open(fp, 'r') as f:
		data = f.read().split('\n')
	step = []
	reward = []
	for d in data:
		if len(d) == 0:
			continue
		j = json.loads(d)
		step.append(j['step'])
		if train:
			key = 'episode_reward'
			# key = 'success_rate'
		elif 'episode_reward_test_env' in j.keys():
			key = 'episode_reward_test_env'
		else:
			key = 'episode_reward_dr'
		reward.append(j[key])
	
	return np.stack([step, reward])


def load_log(fp, train):
	log_file = os.path.join(fp, 'train.log' if train else 'eval.log')
	return log_path_to_numpy(log_file, train)


def load_date(fp):
	with open(os.path.join(fp, 'info.log'), 'r') as f:
		data = f.read().split('\n')
	try:
		j = json.loads(''.join(data))
	except:
		for d in data:
			if len(d) == 0:
				continue
			j = json.loads(d)
			break
	return j['timestamp'].split(' ')[0]


def plot_curve(ax, benchmark, env_name, curve, seeds, train=True, color='black', smoothing=None):
	all_rew = []
	for seed in seeds:
		try:
			fp = os.path.join(benchmark, env_name, curve, str(seed))
			rew = load_log(fp, train)
			if smoothing is not None:
				rew[1] = running_mean(rew[1], smoothing)
			assert len(rew[0]) == len(rew[1]), f'rew shape {rew.shape}'
			if not train and rew[0][-1] < 500_000:
				date = load_date(fp)
				print('Warning: incomplete data for', fp, '| date:', date)
			all_rew.append(rew)
		except:
			if train:
				print('Failed to load', fp)
	if len(all_rew) == 0:
		return
	all_rew = np.concatenate(all_rew, axis=-1)
	df = pd.DataFrame.from_dict(dict(
		steps=all_rew[0]/1e3,
		rewards=all_rew[1],
	))
	sns.lineplot(
		x='steps',
		y='rewards',
		data=df,
		color=color,
		label=curve,
		legend=False,
		ci='sd',
		err_kws={'alpha': 0.1},
		ax=ax,
		linewidth=2,
		dashes=None,
		alpha=1,
		style=True
	)


if __name__ == '__main__':
	utils.make_dir('plots')
	log_dir = 'logs'
	envs = [
		'robot_reach',
		'robot_push',
		'robot_lift',
		'robot_pickplace'
	]
	curves = [
		'sacv2/sh4-proj50-fs3',
    ]
	lims=((0, 300), (-30, 0))

	f, axs = plt.subplots(1, len(envs), sharex='col', figsize=(4*len(envs), 4.4))
	for j, env in enumerate(envs):
		ax = axs[j]
		color_idx = 0
		ax.xaxis.set_ticks(np.arange(0, 501, 100))
		for curve in curves:
			plot_curve(ax, log_dir, env, curve, np.arange(2), train=True, color=color_palette[color_idx], smoothing=100)
			color_idx += 1
		ax.set_title(env, fontsize=16)
		ax.set_ylim(*lims[1])
		if j == 0:
			ax.set_ylabel('Episode return', fontsize=16)
		else:
			ax.set_ylabel('')
			ax.yaxis.set_ticklabels([])
		ax.set_xlabel('Number of frames (' + r'$\times 10^{3}$' + ')', fontsize=16)
		ax.set_xlim(*lims[0])
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(15)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(15)
	h, l = axs[0].get_legend_handles_labels()
	f.legend(h, l, loc='lower center', ncol=len(curves), prop={'size': 18})
	plt.tight_layout()
	f.subplots_adjust(bottom=0.3, wspace=0.125)
	plt.savefig(os.path.join('plots', 'train.png'), dpi=500)
	plt.savefig(os.path.join('plots', 'train.pdf'), dpi=500)
	plt.close()
