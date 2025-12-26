"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
from standalone.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--show_camera", action="store_true", default=False, help="Show camera of one environment.")
parser.add_argument("--use_auxiliary_head", action="store_true", default=False, help="Use auxiliary head for vision policy.")
parser.add_argument(
    "--viz",
    choices={"gui", "camera"},
    default=None,
    help=(
        "Visualization backend. 'gui' uses Isaac Sim viewport (may be slow to start); "
        "'camera' shows the front RayCasterCamera with matplotlib (forces --headless, avoids RTX/viewport). "
        "Default: 'camera' when --show_camera, otherwise 'gui'."
    ),
)
parser.add_argument(
    "--video_backend",
    choices={"gym", "imageio", "opencv"},
    default=None,
    help=(
        "Video recording backend. 'gym' uses Gymnasium RecordVideo with Isaac Sim RGB rendering; "
        "'imageio' records the front RayCasterCamera depth visualization to mp4; "
        "'opencv' is deprecated (alias of 'imageio'). "
        "Default: 'imageio' in --viz camera, otherwise 'gym'."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Resolve defaults for viz/video backends.
if args_cli.viz is None:
    # If the user asked to "show camera", prefer the stable headless RayCaster preview by default.
    if args_cli.show_camera:
        args_cli.viz = "camera"
    else:
        args_cli.viz = "camera" if getattr(args_cli, "headless", False) else "gui"
if args_cli.video_backend is None:
    args_cli.video_backend = "imageio" if args_cli.viz == "camera" else "gym"
elif args_cli.video_backend == "opencv":
    print("[WARN] --video_backend=opencv is deprecated; using --video_backend=imageio instead.")
    args_cli.video_backend = "imageio"

# RGB rendering is only required for gym-based viewport recording.
need_rgb_render = bool(args_cli.video and args_cli.video_backend == "gym")

# In camera viz mode, we intentionally avoid Isaac Sim viewport/RTX to prevent startup stalls.
if args_cli.viz == "camera":
    args_cli.headless = True
    # RayCasterCamera does not require RTX cameras / offscreen rendering. Force-disable by default for stability.
    if not need_rgb_render:
        os.environ["ENABLE_CAMERAS"] = "0"
        args_cli.enable_cameras = False

# Enable RTX cameras / rendering only when we need RGB frames (gym wrapper).
if need_rgb_render:
    args_cli.enable_cameras = True
    if getattr(args_cli, "rendering_mode", None) is None:
        # Bias towards performance to reduce startup/rendering load.
        args_cli.rendering_mode = "performance"

if args_cli.viz == "camera":
    print(
        "[INFO] --viz=camera enabled: forcing --headless and visualizing via depth camera (no Isaac Sim viewport)."
        f" video_backend={args_cli.video_backend}"
    )
else:
    # Force GUI mode even if the user has `HEADLESS=1` in their environment.
    os.environ["HEADLESS"] = "0"
    if args_cli.show_camera:
        print("[INFO] --show_camera: using Isaac Sim GUI viewport (no extra depth preview window).")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from standalone.rsl_rl.ext.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from standalone.rsl_rl.ext.utils import export_vision_policy_as_onnx
from isaaclab.sensors import RayCasterCamera
# Import extensions to set up environment tasks
import diff.lab # noqa: F401
import diff.lab_tasks  # noqa: F401

def _visualize_depth(depth_map: np.ndarray, max_depth: float = 10.0, colored: bool = True) -> np.ndarray:
    """Convert a depth map to a uint8 image (HxWx3 RGB if colored else HxW)."""
    depth_map = np.nan_to_num(depth_map, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth_clipped = np.clip(depth_map, 0.0, max_depth)
    depth_normalized = (depth_clipped / max_depth).astype(np.float32)

    if not colored:
        return (depth_normalized * 255.0).astype(np.uint8)

    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("jet")
    rgba = cmap(depth_normalized)  # HxWx4 float in [0, 1]
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def _overlay_text_rgb(image_rgb: np.ndarray, text: str, xy: tuple[int, int] = (10, 10)) -> np.ndarray:
    """Draw text with a black stroke onto an RGB uint8 image."""
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(pil_img)


def _pad_to_even_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Pad an RGB image so width/height are even (required by many H.264 encoders)."""
    height, width = image_rgb.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return image_rgb
    out = np.zeros((height + pad_h, width + pad_w, 3), dtype=image_rgb.dtype)
    out[:height, :width] = image_rgb
    if pad_h:
        out[height:, :width] = image_rgb[height - 1 : height, :width]
    if pad_w:
        out[:height, width:] = image_rgb[:height, width - 1 : width]
    if pad_h and pad_w:
        out[height:, width:] = image_rgb[height - 1, width - 1]
    return out


def _try_init_matplotlib_viewer():
    """Best-effort live viewer using matplotlib; returns (plt, fig, im) or (None, None, None)."""
    try:
        import matplotlib

        if os.environ.get("DISPLAY"):
            try:
                matplotlib.use("TkAgg", force=True)
            except Exception:
                pass

        import matplotlib.pyplot as plt

        if matplotlib.get_backend().lower() == "agg":
            return None, None, None

        plt.ion()
        fig, ax = plt.subplots(num="Depth (env 0)")
        ax.set_axis_off()
        im = ax.imshow(np.zeros((16, 16, 3), dtype=np.uint8))
        fig.tight_layout(pad=0)
        return plt, fig, im
    except Exception:
        return None, None, None

def _resolve_resume_path(log_root_path, agent_cfg):
    """Resolve checkpoint path, allowing direct file or custom run directory."""
    # Direct checkpoint path provided
    if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint):
        return agent_cfg.load_checkpoint

    # Custom run directory provided (absolute or relative)
    if agent_cfg.load_run and os.path.isdir(agent_cfg.load_run):
        run_path = os.path.abspath(agent_cfg.load_run)
        if agent_cfg.load_checkpoint:
            candidate = os.path.join(run_path, agent_cfg.load_checkpoint)
            if os.path.isfile(candidate):
                return candidate
        # fallback: pick latest *.pt / *.pth in the provided directory
        candidates = [f for f in os.listdir(run_path) if f.endswith(".pt") or f.endswith(".pth")]
        if candidates:
            candidates.sort(key=lambda m: f"{m:0>15}")
            return os.path.join(run_path, candidates[-1])

    # Default: use rsl_rl log layout
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_auxiliary_head:
        resume_path = _resolve_resume_path(log_root_path, agent_cfg)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    use_gym_video = args_cli.video and args_cli.video_backend == "gym"
    use_imageio_video = args_cli.video and args_cli.video_backend == "imageio"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if use_gym_video else None)
    # wrap for video recording
    if use_gym_video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.use_auxiliary_head:
        print("[INFO] Using auxiliary head for vision policy.")
        decoder = ppo_runner.get_decoder()
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    export_vision_policy_as_onnx(
        ppo_runner.alg.policy,
        export_model_dir,
        ppo_runner.obs_normalizer,
        "vision_policy.onnx",
        False,
        env.unwrapped.scene['front_camera'].image_shape,
        (16,),
        use_auxiliary_head=args_cli.use_auxiliary_head,
    )

    # reset environment
    policy_obs = env.get_observations()['policy']
    timestep = 0
    frame_index = 0
    prob = []
    actions_list = []

    # Optional depth video recording (from RayCasterCamera).
    depth_video_writer = None  # imageio Writer (dict wrapper)
    depth_video_path = None
    video_dir = None
    ckpt_tag = None
    fps = float(1.0 / env.unwrapped.step_dt)
    record_depth_video = args_cli.viz == "camera" and use_imageio_video
    need_camera_frames = args_cli.viz == "camera" and (args_cli.show_camera or record_depth_video)
    if need_camera_frames:
        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        ckpt_tag = os.path.splitext(os.path.basename(resume_path))[0]
        if record_depth_video:
            depth_video_path = os.path.join(video_dir, f"depth_{ckpt_tag}.mp4")
            depth_video_writer = {"fps": fps, "writer": None}
    camera: RayCasterCamera | None = None
    mpl_plt = None
    mpl_fig = None
    mpl_im = None
    dump_preview_path = None
    dump_preview_every = 5
    if need_camera_frames:
        camera = env.unwrapped.scene["front_camera"]

        if args_cli.show_camera:
            mpl_plt, mpl_fig, mpl_im = _try_init_matplotlib_viewer()
            if mpl_plt is None:
                dump_preview_path = os.path.join(video_dir or log_dir, "depth_live.png")
                print(
                    "[INFO] Matplotlib GUI backend unavailable in Isaac Sim Python (backend=Agg). "
                    f"Writing a periodically-updated preview image: {dump_preview_path}"
                )
                args_cli.show_camera = False
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, feats = policy(policy_obs)
            if args_cli.use_auxiliary_head:
                prob.append(decoder(feats).sigmoid()[0].item())
            # env stepping
            obs, _, _, _ = env.step(actions)
            actions_list.append(actions[0])
            frame_index += 1
            policy_obs = obs['policy']

        if need_camera_frames and camera is not None:
            depth = camera.data.output["distance_to_image_plane"]
            depth_vis = _visualize_depth(
                depth[0, :, :, 0].cpu().numpy(), max_depth=float(getattr(camera.cfg, "max_distance", 10.0)), colored=True
            )
            if args_cli.use_auxiliary_head and prob:
                depth_vis = _overlay_text_rgb(depth_vis, f"CP: {prob[-1]:.2f}", xy=(10, 10))

            if args_cli.show_camera and mpl_plt is not None and mpl_fig is not None and mpl_im is not None:
                if not mpl_plt.fignum_exists(mpl_fig.number):
                    args_cli.show_camera = False
                else:
                    mpl_im.set_data(depth_vis)
                    mpl_fig.canvas.draw_idle()
                    mpl_plt.pause(0.001)

            if record_depth_video and isinstance(depth_video_writer, dict):
                if depth_video_writer["writer"] is None:
                    depth_video_writer["writer"] = imageio.get_writer(
                        depth_video_path,
                        fps=depth_video_writer["fps"],
                        codec="libx264",
                        quality=8,
                        macro_block_size=None,
                    )
                depth_video_writer["writer"].append_data(_pad_to_even_rgb(depth_vis))

            if dump_preview_path is not None and frame_index % dump_preview_every == 0:
                imageio.imwrite(dump_preview_path, depth_vis)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break
    # close the simulator
    env.close()
    if mpl_plt is not None and mpl_fig is not None:
        with np.errstate(all="ignore"):
            mpl_plt.close(mpl_fig)
    if record_depth_video and isinstance(depth_video_writer, dict) and depth_video_writer["writer"] is not None:
        depth_video_writer["writer"].close()
        if depth_video_path is not None:
            print(f"[INFO] Depth video saved to: {depth_video_path}")
    import matplotlib.pyplot as plt
    plt.ioff()
    # plot the probability of the auxiliary head
    if args_cli.use_auxiliary_head:
        plt.plot(prob)
        plt.xlabel("Timestep")
        plt.ylabel("Probability")
        plt.title("Probability of the auxiliary head")
        plt.savefig(os.path.join(log_dir, "auxiliary_prob.png"))
        # plot actions
        Actions = torch.stack(actions_list)
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot((Actions[:, 0].tanh() * 1.5 * 9.81 + 1.5 * 9.81).cpu().numpy(), label="thrust")
        plt.subplot(4, 1, 2)
        plt.plot((Actions[:, 1].tanh() * 6).cpu().numpy(), label="wx")
        plt.subplot(4, 1, 3)
        plt.plot((Actions[:, 2].tanh() * 6).cpu().numpy(), label="wy")
        plt.subplot(4, 1, 4)
        plt.plot((Actions[:, 3].tanh() * 6).cpu().numpy(), label="wz")
        plt.title("Actions")
        plt.savefig(os.path.join(log_dir, "actions.png"))

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
