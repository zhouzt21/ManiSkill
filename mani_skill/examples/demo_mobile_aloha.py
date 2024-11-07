import gymnasium as gym
import numpy as np
import mplib
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.structs.pose import Pose


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "TestTable-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "mobile_aloha"
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = "none"
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos_vel"
    """Control mode"""

    render_mode: str = "human"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
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

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # while True:
    #     action = env.action_space.sample() if env.action_space is not None else None
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if verbose:
    #         print("reward", reward)
    #         print("terminated", terminated)
    #         print("truncated", truncated)
    #         print("info", info)
    #     if args.render_mode is not None:
    #         env.render()
    #     if args.render_mode is None or args.render_mode != "human":
    #         if (terminated | truncated).any():
    #             break
    # env.agent.robot.set_root_pose(Pose.create_from_pq([0, -0.65, 0], [1, 0, 0, 1]))
    # import pdb; pdb.set_trace()
    planner = mplib.Planner(
        urdf=env.agent.urdf_path,
        srdf=env.agent.srdf_path,
        move_group=env.agent.ee_link_name
    )
    arm_base_link = env.agent.robot.find_link_by_name(
        env.agent.arm_base_link_name
    )
    arm_base_pose = arm_base_link.pose.raw_pose.numpy()[0]
    planner.set_base_pose(arm_base_pose)

    cube_pose = env.cube.pose.raw_pose

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    target_pose = [*cube_pose.numpy()[0,:2], cube_pose[0, 2].item() + 0.3, 1, 0, 0, 0]
    print("arm_pose", arm_base_pose)
    print("target", target_pose)

    plan_to_pose(target_pose, planner, env.agent, env)

    if record_dir:
        print(f"Saving video to {record_dir}")

def plan_to_pose(pose, planner, agent, env):
    qpos = []
    for joint_name in agent.arm_joint_names:
        qpos.append(agent.robot.find_joint_by_name(joint_name).qpos.squeeze())
    # import pdb; pdb.set_trace()
    
    result = planner.plan_screw(
        target_pose=pose,
        qpos=qpos,
        time_step=1 / 250,
        use_point_cloud=False,
        use_attach=False,
    )
    print("plan ",result["status"])
    if result["status"] == "Success":
        n_step = result["position"].shape[0]
        for i in range(n_step):
            qf = agent.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            agent.robot.set_qf(qf)


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
