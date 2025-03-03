import gymnasium as gym
import numpy as np
import sapien
import os

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill import PACKAGE_DIR

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union


def get_dual_arm_data(data_dict, step_id, data_name: str):
    if data_name == "actions" or data_name=="states":
        data_read=[]
        # left
        for joint_id in range(6):
            data_read.append(data_dict[data_name][step_id][0][50+joint_id])
        data_read.append(data_dict[data_name][step_id][0][60])# gripper [0,1]
        # right
        for joint_id in range(6):
            data_read.append(data_dict[data_name][step_id][0][joint_id])
        data_read.append(data_dict[data_name][step_id][0][10])
        #step update
        next_step_id = step_id + 1
        # should be torch.tensor or numpy. tensor better.
        data_read = np.array(data_read, dtype=np.float32)
        return data_read, next_step_id
    else:
        assert 0, "you can just use actions or states as name"


def get_single_arm_data(data_dict, step_id, data_name: str):
    if data_name == "actions" or data_name=="states":
        data_read=[]
        # left: Notice that the left arm is main arm which is right arm in rdt.
        for joint_id in range(6):
            data_read.append(data_dict[data_name][step_id][0][50+joint_id])
        data_read.append(data_dict[data_name][step_id][0][60])# gripper [0,1]
        #step update
        next_step_id = step_id + 1
        # should be torch.tensor or numpy. tensor better.
        data_read = np.array(data_read, dtype=np.float32)
        return data_read, next_step_id
    else:
        assert 0, "you can just use actions or states as name"


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "RoboCasaCustomKitchen-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = 'mobile_aloha'
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = "none"
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "bi_pd_joint_pos" # None
    """Control mode"""

    render_mode: str = "human" # "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 0 # None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=[x + 2022 for x in args.seed], options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # load .npz file
    data_dict = np.load(os.path.join(PACKAGE_DIR,'data/data_0.npz'))
    step = 0
    
    while True:
        '''
            [rdt]
                jonit_pos range     []? Luckily maybe not scaled

                gripper range       [0, 1]? when test find it is not [0,1]

                [order] The index in the state (shape is [128,])
                                -> look up for STATE_VEC_IDX_MAPPING in configs/state_vec.py
                
            [maniskill aloha]

                joint_pos range     [-10,       10]

                gripper range       [-0.01,  0.045]

                [order] The index in the action (shape is [14,]) 
                    arm_left:       joint1  -> 0
                                    jonit2  -> 1
                                    ...
                                    jonit6  -> 5

                    gripper_left:   joint   -> 6

                    arm_right:      joint1  -> 7
                                    jonit2  -> 8
                                    ...
                                    jonit6  -> 12

                    gripper_right:  joint   -> 13
        ''' 
        action_read, step = get_dual_arm_data(data_dict, step, "actions")
        obs, reward, terminated, truncated, info = env.step(action_read)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)# reach max step which is define at register_env function
            print("info", info)
        if args.render_mode is not None:
            env.render()
        # Try to modify
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
        if(step>=data_dict['total_timesteps']):
            break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
