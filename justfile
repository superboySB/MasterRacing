config-git:
    git config --global --add safe.directory /workspace/MasterRacing

train-stage-0:
    TRAINING_STAGE=0 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
    --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless \
    --num_envs 1024 --experiment_name racing --run_name s0

# e.g. --load_run 2025-12-15_17-07-17_s0 --checkpoint model_3999.pt

train-stage-1:
    TRAINING_STAGE=1 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
    --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless --num_envs 1024 \
    --experiment_name racing --run_name s1 --resume True \
    --load_run <stage0_run_dir> --checkpoint <ckpt.pt>


train-stage-2:
    ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.play \
    --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1 --show_camera --resume True \
    --video --video_length 400 --experiment_name racing \
    --load_run <run_dir> --checkpoint <ckpt.pt>
# e.g. --load_run 2025-12-16_01-33-08_s1 --checkpoint model_7998.pt


train-offline-finetune:
    ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.train \
    --save_path logs/offline_finetune --epochs 5 --batch_size 256 --lr 3e-4 \
    --h5_path datasets/racing_stage1.h5 \
    --policy_path logs/rsl_rl/racing/<run_dir>/<ckpt.pt>

# e.g. --policy_path assets/trained_policy/rsl_rl/racing/2025-12-16_01-33-08_s1/model_7998.pt

play-with-demo:
    TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.play_with_demo \
    --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1 --resume True \
    --video --video_length 10000 --show_camera --experiment_name racing --use_auxiliary_head \
    --checkpoint assets/trained_policy/offline_finetune/policy_finetune_epoch-2.pt


play:
    TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.play_with_demo \
    --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1 --resume True \
    --viz gui --experiment_name racing --use_auxiliary_head \
    --rendering_mode performance \
    --checkpoint assets/trained_policy/offline_finetune/policy_finetune_epoch-2.pt