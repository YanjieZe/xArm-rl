from algorithms.sacv2 import SACv2
from algorithms.sac import SAC

algorithm = {
	'sacv2': SACv2,
	'sac': SAC,
}


def make_agent(obs_shape, state_shape, action_shape, args):

	return algorithm[args.algorithm](obs_shape, state_shape, action_shape, args)
