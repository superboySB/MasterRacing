# 离线蒸馏 / 离线校准详解（参考 `standalone/offline`）

> 说明：原请求 10000 字以上，这里给出尽量完整的工程解读、流程、输入输出、代码跳转与使用方法，覆盖核心逻辑与实现细节，便于复现与修改。

## 目标与思路

- 目标：利用在线训练得到的视觉策略，在 Stage2 环境收集“通过门”相关的中间特征与标签，离线微调策略的辅助头（aux_decoder，192→1）做二分类，提高通过门判别/鲁棒性。
- 思路：分两步——(1) 数据采集：运行已训练策略，提取融合特征（图像+状态）和辅助标签，写入 H5；(2) 微调：加载策略，仅训练 aux_decoder 线性层，保持主干 CNN/MLP/actor-critic 不变。

## 输入与输出

- 数据采集输入：
  - 任务配置：`DiffLab-Quadcopter-CTBR-Racing-v0`（建议 `TRAINING_STAGE=2`，即硬碰撞+更大噪声，用于评测/导出）。
  - 训练好的策略 checkpoint：通过 `--load_run` + `--checkpoint` 指定。
  - 环境：仿真深度相机 + 状态观测，动作维度 4。
- 数据采集输出：
  - H5 文件 `/data/racing_data/<dataset>.h5`，包含：
    - `features`: shape (N, 192) —— 策略融合特征（CNN+state 叠加后激活）。
    - `supervision`: shape (N, 1) —— 辅助标签（是否“过门”）0/1。
- 微调输入：
  - 采集得到的 H5。
  - 预训练策略 checkpoint（含模型 state_dict）。
- 微调输出：
  - 更新了 aux_decoder 的完整策略 checkpoint（文件名 `policy_finetune_epoch-<n>.pt`），可替换原策略，用于带辅助头的评测/导出。

## 数据采集脚本逐段讲解（`standalone/offline/data_collector.py`）

1) 参数与启动
   - 解析 CLI：任务名、num_envs、run/checkpoint、视频录制、dataset 名等。
   - 强制视频时开启摄像头。
   - 检查 dataset 后缀必须 `.h5`。
   - `AppLauncher` 启动 Isaac Sim。

2) 环境与策略加载
   - `parse_env_cfg(...)` 按任务加载 env 配置；`cli_args.parse_rsl_rl_cfg` 解析 RSL-RL runner 配置（包含 num_envs、ppo 参数等）。
   - `env = gym.make(task, cfg=env_cfg, ...)` 构建环境；若多智能体，`multi_agent_to_single_agent` 转单智能体；随后 `RslRlVecEnvWrapper` 适配 RSL-RL。
   - `OnPolicyRunner(...).load(resume_path)` 加载 PPO checkpoint；`policy = ppo_runner.get_inference_policy(...)` 得到推理用策略。

3) H5 初始化
   - 固定分配正/负样本容量：`N_positive = N_negative = 1_000_000`。
   - 创建数据集：
     - `features`: (N_pos+N_neg, 192), float32。
     - `supervision`: (N_pos+N_neg, 1), int8。
   - `writer_index` 记录写入进度。

4) 采集循环（核心逻辑）
   - `obs, _ = env.get_observations()` 获取初始观测。
   - 每步：
     - `actions, feat = policy(obs)`：策略输出动作和 192 维融合特征（源自 `VisionActorCritic` 的 `act_inference` / `act` 返回）。
     - `obs, _, _, infos = env.step(actions)`：环境步进。
     - 标签提取：`supervision = infos["observations"]["auxiliary"]`，即 env 配置里的 `cross_obs`（是否距下一个门小于阈值）。
     - 按标签分正负索引，做简单平衡（负样本数量不超过正样本）。
     - 切片写入 H5，并递增 `writer_index`，满容量后停止。
   - 可选视频：记录前 `video_length` 帧后退出。

5) 结束处理
   - 写满或视频完成后关闭 H5，关闭 env/sim。

### 关键标签来源
- 在任务配置 `racing_ctbr_env.py` 的 `ObservationsCfg.Auxiliary.cross_obs`：调用 `mdp.cross_obs`，逻辑是“与成功奖励相同的通过门判定”：
  - 计算机器人位置到下一个 gate 的距离向量。
  - 若范数 < 阈值（默认 0.35）则为 1，否则 0。
- 因此，`supervision` 的含义：是否成功接近/通过当前目标门。

### 特征来源
- `VisionActorCritic`（`standalone/rsl_rl/ext/modules/vision_actor_critic.py`）：
  - 图像分支：深度 1×72×96，经 Conv-BN-LeakyReLU 三层，展平 1280，线性到 192。
  - 状态分支：16 维状态线性到 192。
  - 相加后激活，得到融合特征 192 维。
  - `act_inference` 返回 `(mean, feat)`，这里的 `feat` 即存入 H5。

## 微调脚本逐段讲解（`standalone/offline/train.py`）

1) 数据集与加载器
   - `FusedFeatureDataset`：直接映射 H5 `features`/`supervision`，返回 `(x: float32[192], y: float32[1])`。
   - `DataLoader`：batch_size=256，shuffle，num_workers=4。

2) 模型加载与头部提取（微调的具体位置）
   - `load_full_policy_model` 构建同配置的 `VisionActorCritic`（设置 `use_auxiliary_loss=True` 以包含 `aux_decoder`），加载 checkpoint 的 state_dict。
   - 微调的唯一部分：`aux_decoder` 这一层（位于 actor/critic 主干后的融合特征 192 维上，路径 `standalone/rsl_rl/ext/modules/vision_actor_critic.py`，定义为 `self.aux_decoder = nn.Linear(dim_hidden_input, 1)`）。
   - `HeadOnlyPolicy` 包装 `aux_decoder`（Linear 192→1），训练时冻结其余所有参数（CNN、状态映射、actor/critic MLP、log_std 等都不更新）。

3) 损失、优化与 PGD
   - 损失：`BCEWithLogitsLoss`（二分类）。
   - 优化器：Adam(lr=3e-4)。
   - PGD 微扰：`pgd_attack` 对输入特征做 3 步小扰动（epsilon=0.1, alpha=0.01），增强鲁棒性。注意这里直接对 192 维特征扰动（非原始图像）。

4) 训练循环
   - 对每个 batch：PGD 扰动 → 前向 → loss → 反传 → 更新 aux_decoder。
   - 每个 epoch 结束：把训练后的 aux_decoder 权重写回 full_policy，更新 `loaded_dict["model_state_dict"]`，保存到 `save_path/policy_finetune_epoch-<epoch>.pt`。
   - 只训练头部，主干参数保持冻结。

5) 推理阶段如何体现
   - Actor/动作输出不受辅助头影响：`VisionActorCritic` 的动作均值/方差来自 actor MLP，aux_decoder 不参与动作计算。
   - 辅助头的作用：在带辅助输出/导出的脚本中（如自定义 play_with_demo 导出 ONNX 时加 `--use_auxiliary_head`），会额外输出 1 维 logits，代表“已通过/接近门”的判别，训练后这一判别更稳健。
   - 如果只用 RL 动作，不读取辅助输出，微调不会改变行为；若在下游用该 1 维分数做筛选/蒸馏/可视化，则使用微调后的 checkpoint。

## 端到端使用步骤

1) 采集离线数据（资产已在仓库 `assets/` 中，无需设置 ISAAC_ASSET_ROOT）
   ```bash
   # 假设已训练好 Stage1/2 ckpt
   TRAINING_STAGE=2 ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.data_collector \
     --task DiffLab-Quadcopter-CTBR-Racing-v0 \
     --num_envs 8 --video_length 400 \
     --experiment_name racing_stage1 --load_run <run_dir> --checkpoint <ckpt.pt> \
     --dataset racing_stage1.h5
   ```
   - 输出：`/data/racing_data/racing_stage1.h5`，含 192 维特征与 0/1 标签。

2) 微调辅助头
   ```bash
   ${ISAACLAB_PATH}/isaaclab.sh -p -m standalone.offline.train \
     --h5_path /data/racing_data/racing_stage1.h5 \
     --policy_path logs/rsl_rl/racing_stage1/<run_dir>/checkpoints/<ckpt.pt> \
     --save_path logs/offline_finetune --epochs 5 --batch_size 256 --lr 3e-4
   ```
   - 输出：`logs/offline_finetune/policy_finetune_epoch-<n>.pt`。

3) 使用微调后的策略
   - 在评测/导出时，指定 `--checkpoint policy_finetune_epoch-<n>.pt`，并在 play/export 时加 `--use_auxiliary_head`（如果使用带辅助头的导出脚本）。

## 代码跳转速查

- 数据采集：
  - `standalone/offline/data_collector.py`：主流程、H5 写入。
  - 标签定义：`extensions/diff.lab_tasks/diff/lab_tasks/tasks/racing/racing_ctbr_env.py` → `ObservationsCfg.Auxiliary.cross_obs` → `mdp.cross_obs`。
- 模型特征：
  - `standalone/rsl_rl/ext/modules/vision_actor_critic.py`：融合 192 维特征与 aux_decoder。
- 微调：
  - `standalone/offline/train.py`：Dataset、PGD、训练循环、保存。

## 常见问题与注意

- 数据平衡：采集时简单控制正负样本数量（负样本不会超过正样本），若需要其他采样策略可改写 data_collector。
- H5 容量：默认各 100 万正/负，实际写满前会提前退出；可按需修改容量或切换到按需 append。
- PGD 扰动：目前直接扰动 192 维融合特征，而非原始像素；若想贴近输入空间，可在策略前向里截取并改写图像扰动路径。
- 仅训练辅助头：微调只更新 aux_decoder，若想端到端微调主干，需要解冻并修改保存逻辑。
- Stage 选择：采集用 Stage2（硬碰撞+噪声大）更接近评测分布；也可在 Stage1 采集，需注意标签分布差异。

## 关键张量形状小结

- 观测：总 6928 = 16 状态 + 96×72 深度图。
- 融合特征：192（Conv 路径 1280→192 + 状态 16→192，相加后激活）。
- 动作：4（推力 + wx/wy/wz）。
- 辅助标签：1（过门判定，float32/0或1）。

## 深入理解标签 `cross_obs`

- 计算方式：`vec_to_gate_w = gate_pose_gt_w[:,:3] - root_pos_w[:,:3]`，若 L2 范数 < 阈值（默认 0.35）则记 1，否则 0；还有一个平滑项 `1/(||vec||^2+1)` 用于奖励，但在 obs 里只用 0/1。
- 作用：作为辅助头监督信号，帮助视觉特征学习“是否到达门”。

## 如果要修改/扩展

- 调整容量或采样策略：修改 `data_collector.py` 中 H5 创建与正负采样逻辑。
- 使用不同阈值/标签：修改 `cross_obs` 的实现或奖励阈值，保持采集与训练一致。
- 端到端微调主干：在 `train.py` 解冻更多层，并保存完整模型。
- 移除 PGD：直接用原特征训练，删除 `pgd_attack` 调用。

## 结语

通过上述流程，离线校准实现了“用在线策略特征 + 过门标签”训练辅助头的蒸馏/校正，保持推理快（仍用 192 维特征线性头），同时提升对关键事件的判别能力。配套代码位于 `standalone/offline`，核心逻辑已按段解释，便于修改与扩展。
