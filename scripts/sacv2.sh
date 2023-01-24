# bash scripts/sacv2.sh robot lift xyzw 0 0629 0

# export MUJOCO_GL=egl
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/yanjieze/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P /root/.mujoco/

# tar -xvf /root/.mujoco/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/

# pip uninstall -y metaworld

# pip install git+https://github.com/YanjieZe/metaworld.git@rl3d


use_wandb=0
remove_addition_log=0
domain_name=${1}
env=${2}
action_space=${3}
observation_type=state+image
seed=${4}
date=${5}
save_model_in_wandb=${6}
train_steps=3m


log_train_video=100k

wandb_project="${domain_name}_${env}"


wandb_group="sac-save${save_model_in_wandb}-${date}"

CUDA_VISIBLE_DEVICES=5 python src/train.py \
	--algorithm sacv2 \
	--domain_name ${domain_name} \
	--observation_type  ${observation_type} \
	--task_name ${env} \
	--action_space ${action_space} \
	--episode_length 50 \
    --train_steps ${train_steps}\
	--buffer_capacity 100k \
	--episode_length 50 \
	--exp_suffix default \
	--seed ${seed} \
	--wandb_project ${wandb_project} \
	--wandb_group ${wandb_group} \
	--use_wandb ${use_wandb} \
	--remove_addition_log ${remove_addition_log} \
	--log_train_video ${log_train_video}  \
	--save_model_in_wandb ${save_model_in_wandb} \


