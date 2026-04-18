# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 base_dp3 test 0 0 eef



DEBUG=False
save_ckpt=False
wandb_mode=online

policy_name=${1}
task_name=${2}
config_name=${policy_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${policy_name}-${addition_info}
run_dir="../outputs/${exp_name}_seed${seed}"

gpu_id=${5}
action_mode=${6:-${UR5E_ACTION_MODE:-eef}}


cd dp-family

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export TORCH_HOME=${TORCH_HOME:-/tmp/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache}
export R3M_HOME=${R3M_HOME:-/tmp}
export UR5E_ACTION_MODE=${action_mode}
export UR5E_IS_BIMANUAL=${UR5E_IS_BIMANUAL:-false}
if [ "${action_mode}" = "joint" ]; then
    export UR5E_AGENT_POS_KEY=robot_joint
    export UR5E_ACTION_KEY=joint_action
else
    export UR5E_AGENT_POS_KEY=robot_eef_pose
    export UR5E_ACTION_KEY=cartesian_action
fi
is_bimanual_normalized=$(printf '%s' "${UR5E_IS_BIMANUAL}" | tr '[:upper:]' '[:lower:]')
if [[ "${is_bimanual_normalized}" == "1" || "${is_bimanual_normalized}" == "true" || "${is_bimanual_normalized}" == "yes" || "${is_bimanual_normalized}" == "on" ]]; then
    : "${UR5E_AGENT_POS_DIM:=14}"
    : "${UR5E_ACTION_DIM:=14}"
else
    : "${UR5E_AGENT_POS_DIM:=7}"
    : "${UR5E_ACTION_DIM:=7}"
fi
export UR5E_AGENT_POS_DIM
export UR5E_ACTION_DIM
repo_root=$(pwd)/..
if [[ "${task_name}" == *"image"* || "${task_name}" == "base_dp" ]]; then
    : "${UR5E_RAW_ZARR_PATH:=${repo_root}/data/ur5e_raw.zarr}"
    export UR5E_RAW_ZARR_PATH
elif [[ "${task_name}" == *"idp3"* || "${policy_name}" == *"idp3"* ]]; then
    : "${UR5E_IDP3_ZARR_PATH:=${repo_root}/data/state_idp3.zarr}"
    export UR5E_IDP3_ZARR_PATH
else
    : "${UR5E_ZARR_PATH:=${repo_root}/data/state.zarr}"
    export UR5E_ZARR_PATH
fi
python run_robot.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}
