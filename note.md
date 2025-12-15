# MasterRacing 使用笔记

## 环境与安装
修改api用法，可在zuanfeng的原版容器里使用
```bash
git clone https://github.com/superboySB/MasterRacing
cd MasterRacing && bash install.sh
```
VS Code 补全（容器内代码目录 `/workspace/MasterRacing`，默认 ISAACLAB_PATH=/workspace/isaaclab）：
```bash
cd /workspace/MasterRacing
ISAACLAB_PATH=/workspace/isaaclab ${ISAACLAB_PATH}/isaaclab.sh -p .vscode/tools/setup_vscode.py --isaac_path /workspace/isaaclab
```
脚本会更新 `.vscode/settings.json` 的补全/索引路径，容器内直接用 VS Code Remote / Dev Containers 打开即可。

## 阶段流程概览
- 任务入口 `DiffLab-Quadcopter-CTBR-Racing-v0`，`TRAINING_STAGE` 控制模式。
- 顺序：Stage0 软碰撞 → Stage1 硬碰撞 → （可选）离线校准 → Stage2 评测/导出，再做 sim2real。
- 赛道由 `RacingComplexTerrainCfg` 生成 zigzag/circular/ellipse，每段 8 门，课程按过门数调地形等级与指令噪声。

## 模型与观测简表
- 策略：VisionActorCritic（`standalone/rsl_rl/ext/modules/vision_actor_critic.py`），深度图 96×72 经 3 层卷积+BN 后与状态映射相加成 192 维，再接 actor/critic MLP（128/128，`rsl_rl_ppo_cfg.py`），LeakyReLU，动作噪声可学习。
- 动作 4 维（推力+wx/wy/wz），先 tanh，再在 `DiffActions.process_actions` 中按 `action_scale`/`action_offset` 缩放；推力额外乘质量误差噪声，延迟 0.03s。
- 观测 Stage0/1/2：Policy 6928 维（含噪声），Critic 6928 维（无噪声），Auxiliary 1 维 `cross_obs`。

## 各阶段设置
- **Stage0 软碰撞（TRAINING_STAGE=0）**：禁用碰撞，指令无噪声，episode 6s，仅 z 越界终止。奖励：进度余弦，指令角速度罚 -0.02，动作平滑罚 -0.01，软碰撞罚 -50，感知对齐 0.1，过门 10。
- **Stage1 硬碰撞（TRAINING_STAGE=1）**：碰撞+接触开启，指令噪声递增，episode 6s，含碰撞/翻转终止。奖励：进度与感知同上，指令角速度罚 -0.1，动作平滑罚 -0.05，碰撞 -100，姿态 -30，过门 20。
- **离线校准**：用 Stage1 策略在 Stage2 环境滚动生成 h5（写到 `/data/racing_data/<dataset>`，特征 192，标签 1）。`standalone/offline/train.py` 微调策略的 `aux_decoder` 做二分类，PGD，对应在线设置 `num_actor_obs=6928`、`num_actions=4`。
- **Stage2 评测/导出（TRAINING_STAGE=2）**：与 Stage1 相同碰撞，噪声更大，episode 8s，无姿态惩罚；用于测试与导出。

## 训练与评测命令（按顺序执行）
注意stage 0的1024个env就需要近13G显存

### Stage0 软碰撞
```bash
TRAINING_STAGE=0 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless \
  --num_envs 2048 --experiment_name racing_stage0 --run_name s0
```
### Stage1 硬碰撞（可从 Stage0 恢复）
```bash
TRAINING_STAGE=1 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless --num_envs 2048 \
  --experiment_name racing_stage1 --run_name s1 \
  --resume True --load_run <stage0_run_dir> --checkpoint <ckpt.pt>
```
### Stage2 环境下采集离线数据（用 Stage1 ckpt）
```sh
TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.data_collector \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 8 --video_length 400 \
  --experiment_name racing_stage1 --load_run <run_dir> --checkpoint <ckpt.pt> \
  --dataset racing_stage1.h5
```
### 离线微调辅助头
```sh
${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.train \
  --h5_path /data/racing_data/racing_stage1.h5 \
  --policy_path logs/rsl_rl/racing_stage1/<run_dir>/checkpoints/<ckpt.pt> \
  --save_path logs/offline_finetune --epochs 5 --batch_size 256 --lr 3e-4
```
### Stage2 评测/导出（可展示深度或导出 ONNX）
```sh
TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.play \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1 --show_camera \
  --experiment_name racing_stage1 --load_run <run_dir> --checkpoint <ckpt.pt> \
  --video --video_length 400
```
如需导出带辅助头的 ONNX，用 play_with_demo.py 加 --use_auxiliary_head。如需采集数据，先确保 `mkdir -p /data/racing_data`。

## 常见坑
- 避免混用系统 python，与 Isaac Lab 交互统一用 `${ISAACLAB_PATH}/isaaclab.sh -p ...`。
- OpenCV 缺 Qt 插件时设 `export QT_QPA_PLATFORM_PLUGIN_PATH=${ISAACLAB_PATH}/_isaac_sim/exts/omni.pip.compute/pip_prebundle/cv2/qt/plugins/platforms`。
- VS Code 索引缺失时在 `.vscode/settings.json` 的 `python.analysis.extraPaths` 加上 `extensions` 路径或屏蔽无关 Omniverse 包。