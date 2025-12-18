config-git:
    git config --global --add safe.directory /workspace/MasterRacing

train:
    ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
    --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless \
    --num_envs 128 --experiment_name racing --run_name s0
