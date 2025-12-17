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
- 赛道由 `RacingComplexTerrainCfg` 生成 zigzag/circular/ellipse，每段 8 门；地形 curriculum 按累计过门数 ≥3 升级、<2 降级（全阶段保留）；命令噪声 curriculum 仅 Stage1 启用（初始 ±0.1m / ±0.1rad_yaw，过门≥4 放大 1%，<3 缩小 2%），Stage0 命令无噪声，Stage2 噪声固定 ±0.5m / ±0.5rad_yaw。

## 模型与观测简表
- 策略：VisionActorCritic（`standalone/rsl_rl/ext/modules/vision_actor_critic.py`），无 RNN，支持辅助头。
  - 图像分支：Conv2d 1→16, 3×3, stride 3 → Conv2d 16→32, 3×3, stride 3 → Conv2d 32→64, 2×2, stride 2，三层各接 BN + LeakyReLU；输出 4×5×64=1280，线性到 192。
  - 状态分支：16 维线性到 192；两支相加后 LeakyReLU 得 192 维；Actor/Critic MLP 128→128（`rsl_rl_ppo_cfg.py`），输出均值；动作噪声可学习（scalar 或 log 标准差），可选辅助头把 192 维映射到 1。
- 观测 6928 维 = 深度图 96×72=6912 + 状态 16；Policy/Critic 维度一致，Critic 关闭噪声和命令扰动。
  - base_lin_vel（3）：机体坐标系线速度，policy 乘 (1+0.03·N(0,1))。
  - base_orientation_r（3）：机体 z 轴在世界坐标系的方向向量，policy 加 0.05 rad 高斯欧拉扰动后转矩阵并取第 3 行。
  - target_cmd（6）：`RacingCommand` 的 gate 命令，分别是“机体→当前门”向量(3) 和 “当前门→下一门”向量(3)，都在机体坐标系；policy 读含噪声的 command，critic 用 GT command。
  - last_action（4）：上一步 CTBR 命令，`raw_action.tanh()*scale+offset`，推力项除以 `_robot_mass` 得到 a_zb；观测里不叠加推力估计噪声（噪声只在下发动作时再乘以约 1±2%）。
  - image（6912）：前向深度，裁剪 >10m 置 10m，再 /10 归一化；policy 乘 (1+0.02·N(0,1))，critic 无噪声。
  - Auxiliary：`cross_obs`=1 表示上一步 success_cross>0，用于离线标签。
- 动作 4 维（推力 + wx/wy/wz）：Actor tanh 后经 `DiffActions.process_actions` 缩放。推力范围约 [0, 3g]，体速界限 [-6,6] rad/s；推力附加约 1~2% 估计误差，CTBR 控制延迟 0.03s，行动延迟 1 个 sim 步。
- 奖励项（统一定义，系数分阶段见下文）：
  - progress_rewards：`cos(v_b, vec_gate_gt_b)`，鼓励速度朝向当前门。
  - command_bodyrate_penalty：`-w*||tanh(action[1:4])*max_omega||_2`，惩罚命令角速度幅值。
  - action_rate：`-w*sum((ctbr_t-ctbr_{t-1})^2)`，惩罚命令抖动（推力+角速）。
  - collision_penalty：Stage0 用射线计数>2 记 1，其余阶段用接触传感器>1 记 1。
  - perception_reward：`cos(vec_gate_gt_b, [1,0,0])`，鼓励前向对齐。
  - success_cross：门中心距离 <0.35m 给 `1/(d^2+1)`，否则 0。
  - bad_pose_penalty：仅 Stage1，俯仰或横滚超 90° 记 1。

## 各阶段设置
- **Stage0 软碰撞（TRAINING_STAGE=0）**：用无碰撞 mesh，命令不加噪声；终止只看 z 越界(0~10m)；奖励系数：progress 1、command_bodyrate -0.02、action_rate -0.01、collision_penalty -50（射线近障）、perception 0.1、success_cross 10。episode 6s，terrain curriculum 生效，命令噪声 curriculum 关闭。
- **Stage1 硬碰撞（TRAINING_STAGE=1）**：带 collider+接触传感器，命令噪声作用在当前位置/下一门（roll/pitch 0，yaw ±0.1rad，位置 ±0.1m），过门≥4 放大 1%、<3 缩小 2%；终止：非法接触或翻转/俯仰>90°；奖励：progress 1、command_bodyrate -0.1、action_rate -0.05、collision_penalty -100（真实接触）、perception 0.1、success_cross 20、bad_pose_penalty -30。episode 6s，terrain+命令噪声 curriculum 都开启。
- **离线校准（辅助头，可选）**：在 Stage2 环境用 Stage1 策略滚动，只抽取特征和标签，不改控制逻辑。`data_collector` 生成 192 维融合特征 + `cross_obs` 二值标签（成功过门），`offline/train.py` 仅微调策略的 `aux_decoder` 二分类头（PGD 防御），不动 Actor/Critic；输出 checkpoint 里附带更新后的 aux 头，可在 `play_with_demo --use_auxiliary_head` 或导出 ONNX 时输出“过门置信度”。如果直接用 Stage0+Stage1 在线策略在 Stage2 部署，控制行为已完整；离线校准只提供额外置信度信号，便于 sim2real 过滤/可视化，不影响动作。
- **Stage2 评测/导出（TRAINING_STAGE=2）**：与 Stage1 相同碰撞/终止，命令噪声固定作用在位置 ±0.5m、yaw ±0.5rad（roll/pitch 0），命令噪声 curriculum 关闭但 terrain curriculum 仍在；奖励沿用 Stage1 但无 bad_pose_penalty（依然因姿态终止）；success_cross 20，episode 8s，用于评测、导出和数据收集（可选择是否加载离线微调过的辅助头以输出置信度）。

## 训练与评测命令（按顺序执行）
资产已经放在仓库的 `assets/` 里（UIElements、PolyHaven 天空贴图、vMaterials_2/Ground/Asphalt_Fine.mdl），运行下面的命令无需再设置 `ISAAC_ASSET_ROOT` 或联网
```sh
export OMNI_DATASTORE_ENABLED=0 && \
export OMNI_KIT_DISABLE_TELEMETRY=1 && \
export OMNI_KIT_DISABLE_CRASH_REPORTING=1 && \
export CARB_DISABLE_PYTHON_USDPREVIEW=1
```
注意stage 0/1大约1024个env就需要近13G显存, 2048个env需要近23G显存，4096个env就需要近37G显存。黑盒角度来看，并不是每个阶段的环境数多于2048以后都收益明显,每个阶段训完的`log`应该整个目录都暂时更名放在了`assets/trained_policy`作为快照，必要时可以拖回到cwd

### Stage0 软碰撞
```bash
CUDA_VISIBLE_DEVICES=7 TRAINING_STAGE=0 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless \
  --num_envs 2048 --experiment_name racing --run_name s0
```
预计7小时左右。

### Stage1 硬碰撞
用 Stage0 ckpt继续
```bash
CUDA_VISIBLE_DEVICES=7 TRAINING_STAGE=1 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.train \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --headless --num_envs 2048 \
  --experiment_name racing --run_name s1 --resume True \
  --load_run <stage0_run_dir> --checkpoint <ckpt.pt>
```
举个例子: `--load_run 2025-12-15_17-07-17_s0 --checkpoint model_3999.pt`，预计5小时左右。

### Stage2 环境下trained policy交互
用 Stage1 ckpt继续，可以先本地验证play
```sh
CUDA_VISIBLE_DEVICES=0 TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.play \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1 --show_camera --resume True \
  --video --video_length 400 --experiment_name racing \
  --load_run <run_dir> --checkpoint <ckpt.pt>
```
示例：`--load_run 2025-12-16_01-33-08_s1 --checkpoint model_7998.pt`。

说明：
- 由于部分机器上 Isaac Sim GUI 在 `Starting the simulation...` 阶段可能卡死，这里的 `play/play_with_demo` 在检测到 `--show_camera` 时会默认切到稳定的 `--viz=camera`（强制 `--headless`），不再依赖 GUI。
- 录制输出默认写到：`logs/rsl_rl/racing/<run_dir>/videos/play/`：
  - `depth_<ckpt>.mp4`：RayCaster 前向深度可视化（分辨率 96×72，与策略网络输入一致）。
  - 若 Matplotlib 在 Isaac Sim Python 内无 GUI backend，`--show_camera` 不会弹窗，而是周期性更新 `depth_live.png` 作为预览。
- ONNX 导出在：`logs/rsl_rl/racing/<run_dir>/exported/vision_policy.onnx`。

如必须看 Isaac Sim GUI，可显式加 `--viz gui`（但在部分环境下可能仍会卡死）。如需录制渲染 RGB，可改用 `--video_backend gym --enable_cameras`（更重，可能更容易卡死）。

如果上面没问题了，然后再在服务器开始收集数据给后续的辅助任务
```sh
CUDA_VISIBLE_DEVICES=7 TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.data_collector \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 2048 --video_length 400 --headless \
  --experiment_name racing --dataset racing_stage1.h5 \
  --load_run <run_dir> --checkpoint <ckpt.pt>
```
示例：`--load_run 2025-12-16_01-33-08_s1 --checkpoint model_7998.pt`，预计2小时左右，相应关键tensor（包含状态表征和label）会保存为`datasets/racing_stage1.h5`

### Sim2Real辅助任务
离线微调辅助头
```sh
CUDA_VISIBLE_DEVICES=0 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.train \
  --save_path logs/offline_finetune --epochs 5 --batch_size 256 --lr 3e-4 \
  --h5_path datasets/racing_stage1.h5 \
  --policy_path logs/rsl_rl/racing/<run_dir>/<ckpt.pt>
```
示例：`--policy_path assets/trained_policy/rsl_rl/racing/2025-12-16_01-33-08_s1/model_7998.pt`，预计1小时左右.

### 最终验证
本机用微调后的策略在 Stage2 下评测/录制，可选辅助头输出。
```sh
CUDA_VISIBLE_DEVICES=0 TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.rsl_rl.play_with_demo \
  --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1 --resume True \
  --video --video_length 400 --show_camera --experiment_name racing --use_auxiliary_head \
  --load_run <run_dir> --checkpoint policy_finetune_epoch-xxx.pt
```
视频保存在 `logs/rsl_rl/racing/<run_dir>/videos/play/`（默认生成 `depth_*.mp4`）；只想离屏评测时，可去掉 `--show_camera` 或显式加 `--headless`。

若用了离线微调辅助头，则checkpoint设置的最后一行可以灵活处理，比如：`--checkpoint assets/trained_policy/offline_finetune/policy_finetune_epoch-2.pt`

若用未offline微调策略，去掉 `--use_auxiliary_head`，checkpoint 换成在线 PPO 的 `model_xxxx.pt`，比如`--load_run 2025-12-16_01-33-08_s1 --checkpoint model_7998.pt`。

`--use_auxiliary_head` 额外输出说明：

- `auxiliary_prob.png`：辅助头 `sigmoid(aux_decoder(feats))` 的时间序列（横轴 timestep，纵轴概率），训练标签是 `cross_obs`（成功过门事件的二值信号）；曲线尖峰接近 1 时表示模型对“发生过门事件”的置信度很高。
- `actions.png`：4 个子图依次是推力、wx、wy、wz 的动作时间序列（与训练时一致的 `tanh+scale` 映射）；推力约在 `[0, 3g]`，角速度约在 `[-6, 6] rad/s`，接近边界/突变通常代表更激进的操控或动作饱和。

这两个文件会保存到当前 `--checkpoint` 文件所在目录（例如 `assets/trained_policy/offline_finetune/`），而不是 `logs/rsl_rl/...` 的 run 目录。

<!-- ## 常见坑
- 避免混用系统 python，与 Isaac Lab 交互统一用 `${ISAACLAB_PATH}/isaaclab.sh -p ...`。
- OpenCV 缺 Qt 插件时设 `export QT_QPA_PLATFORM_PLUGIN_PATH=${ISAACLAB_PATH}/_isaac_sim/exts/omni.pip.compute/pip_prebundle/cv2/qt/plugins/platforms`。
- VS Code 索引缺失时在 `.vscode/settings.json` 的 `python.analysis.extraPaths` 加上 `extensions` 路径或屏蔽无关 Omniverse 包。 -->
