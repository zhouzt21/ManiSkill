import multiprocessing as mp
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Optional
import tqdm

import gymnasium as gym
import h5py
import numpy as np
import tyro

import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers

@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    num_procs: int = 1
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing
    and only works with the CPU simulation backend at the moment."""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: str = "default"
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing
    and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: int = 30
    """The FPS of saved videos"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""

    num_envs: Optional[int] = None
    """Number of environments to run to replay trajectories # TODO ELABORATE."""


def parse_args(args=None):
    return tyro.cli(Args, args=args)

def _process_single_file(args, traj_path, proc_id: int = 0, num_procs=1):
    """Process a single H5 file."""
    mode = "cut"
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    try:
        ori_h5_file = h5py.File(traj_path, "r")
    except Exception as e:
        print(f"Error opening {traj_path}: {e}")
        return None

    # test origin
    if mode == "origin":
        env_seed = os.path.basename(os.path.dirname(os.path.dirname(traj_path)))
    # test cut
    elif mode == "cut":
        env_seed = os.path.splitext(os.path.basename(traj_path))[0].split('_')[-1]
    env_seed = int(env_seed)
    print("env_seed:", env_seed)
    print("type(env_seed):", type(env_seed))

    # Load associated json
    json_path = "../collect_data/vr_teleop_data/env/trajectory.json"  
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    # Create a twin env with the original kwargs
    if args.target_control_mode is not None:
        if args.sim_backend:
            ori_env_kwargs["sim_backend"] = args.sim_backend
        ori_env = gym.make(env_id, **ori_env_kwargs)
    else:
        ori_env = None

    # Create a main env for replay
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    env_kwargs = ori_env_kwargs.copy()
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    env_kwargs["shader_dir"] = args.shader
    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs["render_mode"] = (args.render_mode)  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    if args.num_envs is not None:
        env_kwargs["num_envs"] = args.num_envs
    
    if args.sim_backend:
        env_kwargs["sim_backend"] = args.sim_backend
    env = gym.make(env_id, **env_kwargs)
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )
    
    # Prepare for recording
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    suffix = "{}.{}.{}".format(env.obs_mode, env.control_mode, env.device.type)
    new_traj_name = ori_traj_name + "." + suffix

    env = wrappers.RecordEpisode(
        env,
        output_dir=f"../collect_data/vr_teleop_data/data_process/joint_pos_cut",  #f"./output_dir",
        save_trajectory=args.save_traj,
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        video_fps=args.video_fps,
        record_reward=args.record_rewards,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None
    
    episodes = json_data["episodes"][: args.count]

    #----------- Replay-----------------
    ep = episodes[0]
    episode_id = ep["episode_id"]
    traj_id = f"traj_{episode_id}"
    if pbar is not None:
        pbar.set_description(f"Replaying {traj_id}")

    reset_kwargs = ep["reset_kwargs"].copy()
    reset_kwargs.pop("seed")
    seed = env_seed

    ori_control_mode = ep["control_mode"]

    # Each trial for each trajectory to replay, we reset the environment
    # and optionally set the first environment state
    env.reset(seed=seed, **reset_kwargs)
    if ori_env is not None:
        ori_env.reset(seed=seed, **reset_kwargs)

    # Original actions to replay
    if mode == "origin":
        ori_actions = ori_h5_file[traj_id]["actions"][:]
    elif mode == "cut":    
        ori_actions = ori_h5_file["action"][:]

    assert (
        target_control_mode is None
        or ori_control_mode == target_control_mode
        or not args.use_env_states
    ), "Cannot use env states when trying to \
        convert from one control mode to another. This is because control mode conversion causes there to be changes \
        in how many actions are taken to achieve the same states"

    if target_control_mode is None or ori_control_mode == target_control_mode:
        n = len(ori_actions)
        if pbar is not None:
            pbar.reset(total=n)
        for t, a in enumerate(ori_actions):
            if pbar is not None:
                pbar.update()
            _, _, _, truncated, info = env.step(a)
            if args.vis:
                env.base_env.render_human()
    elif ori_control_mode == "bi_pd_ee_target_delta_pose" and target_control_mode == "bi_pd_joint_pos": 
         # TODO: add more control modes,  bi_pd_ee_target_delta_pose --> bi_pd_joint_pos
        # Get the original and target controllers
        print("bi_pd_ee_target_delta_pose --> bi_pd_joint_pos")
        ori_controller = ori_env.agent.controller 
        controller = env.agent.controller

        n = len(ori_actions)
        if pbar is not None:
            pbar.reset(total=n)

        info = {}

        for t in range(n):
            if pbar is not None:
                pbar.update()

            # Split the original action into components for each arm
            ori_action = ori_actions[t]
            ori_action_dict = ori_controller.to_action_dict(ori_action)
            
            # Create output action dictionary while preserving gripper actions
            output_action_dict = {
                'arm_left': None,
                'arm_right': None,
                'gripper_left': ori_action_dict['gripper_left'],
                'gripper_right': ori_action_dict['gripper_right']
            }

            # Apply original action to get target poses
            ori_env.step(ori_action)

            # Handle each arm separately
            for arm_name in ['arm_left', 'arm_right']:
                # Get current joint positions as target
                target_qpos = ori_controller.controllers[arm_name].qpos
                target_qpos = target_qpos.squeeze_()
                output_action_dict[arm_name] = target_qpos

            # Combine and execute the converted action
            output_action = controller.from_action_dict(
                common.to_tensor(output_action_dict, device=env.device)
            )
            _, _, _, _, info = env.step(output_action)

            if args.vis:
                env.base_env.render_human()
    else:
        raise NotImplementedError(f"Control mode {ori_control_mode} to {target_control_mode} conversion is not supported")


    if args.save_traj:
        env.flush_trajectory()
    if args.save_video:
        env.flush_video(name =f"env_{env_seed}",ignore_empty_transition=False)

    # Cleanup
    env.close()
    ori_h5_file.close()

    if pbar is not None:
        pbar.close()

    return output_h5_path

def main(args):
    # Specify the directory containing the H5 files
    directory = "../collect_data/vr_teleop_data/data_process/cut/h5"  

    # Get a list of all H5 files in the directory
    h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]

    # Process each file
    for i, h5_file in enumerate(h5_files):
        print(f"Processing file: {h5_file} ({i+1}/{len(h5_files)})")
        args.traj_path = h5_file  # Update traj_path in args for each file
        _process_single_file(args, h5_file, proc_id=0, num_procs=1)  # Process the file

if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())