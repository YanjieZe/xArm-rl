from gym.envs.registration import register

REGISTERED_ROBOT_ENVS = False


def register_robot_envs(n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False):
	global REGISTERED_ROBOT_ENVS
	if REGISTERED_ROBOT_ENVS:	
		return

	register(
		id='RobotLift-v0',
		entry_point='env.robot.lift:LiftEnv',
		kwargs=dict(
			xml_path='robot/lift.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotLifteasy-v0',
		entry_point='env.robot.lift_easy:LiftEnv',
		kwargs=dict(
			xml_path='robot/lift.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPickplace-v0',
		entry_point='env.robot.pick_place:PickPlaceEnv',
		kwargs=dict(
			xml_path='robot/pick_place.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPegbox-v0',
		entry_point='env.robot.peg_in_box:PegBoxEnv',
		kwargs=dict(
			xml_path='robot/peg_in_box.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
			
		)
	)
	
	register(
		id='RobotDrawer-v0',
		entry_point='env.robot.drawer:DrawerEnv',
		kwargs=dict(
			xml_path='robot/drawer_open.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotDrawerclose-v0',
		entry_point='env.robot.drawer_close:DrawerCloseEnv',
		kwargs=dict(
			xml_path='robot/drawer.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotHammer-v0',
		entry_point='env.robot.hammer:HammerEnv',
		kwargs=dict(
			xml_path='robot/hammer.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotHammerall-v0',
		entry_point='env.robot.hammer_all:HammerAllEnv',
		kwargs=dict(
			xml_path='robot/hammer_all.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotReach-v0',
		entry_point='env.robot.reach:ReachEnv',
		kwargs=dict(
			xml_path='robot/reach.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotReachmovingtarget-v0',
		entry_point='env.robot.reach:ReachMovingTargetEnv',
		kwargs=dict(
			xml_path='robot/reach.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPush-v0',
		entry_point='env.robot.push:PushEnv',
		kwargs=dict(
			xml_path='robot/push.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPushnogoal-v0',
		entry_point='env.robot.push:PushNoGoalEnv',
		kwargs=dict(
			xml_path='robot/push.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)



	register(
		id='RobotShelfplacing-v0',
		entry_point='env.robot.shelf_placing:ShelfPlacingEnv',
		kwargs=dict(
			xml_path='robot/shelf_placing.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	

	

	register(
		id='RobotReachwall-v0',
		entry_point='env.robot.reach_wall:ReachWallEnv',
		kwargs=dict(
			xml_path='robot/reach_wall.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# pull up the lever
	register(
		id='RobotLeverpull-v0',
		entry_point='env.robot.lever_pull:LeverPullEnv',
		kwargs=dict(
			xml_path='robot/lever_pull.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# peg insert side
	register(
		id='RobotPeginsert-v0',
		entry_point='env.robot.peg_insert:peginsertEnv',
		kwargs=dict(
			xml_path='robot/peg_insert.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# align real and sim
	register(
		id='RobotPeginsertreal-v0',
		entry_point='env.robot.peg_insert_side:peginsertEnv',
		kwargs=dict(
			xml_path='robot/peg_insert_side_real.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# hammer pick
	register(
		id='RobotHammerpick-v0',
		entry_point='env.robot.hammer_pick:HammerPickEnv',
		kwargs=dict(
			xml_path='robot/hammer_pick.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)


	# shelf task in real
	register(
		id='RobotShelfreal-v0',
		entry_point='env.robot.shelf_real:ShelfPlacingEnv',
		kwargs=dict(
			xml_path='robot/shelf_real.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)


	

	REGISTERED_ROBOT_ENVS = True
