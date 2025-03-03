from pathlib import Path

import sapien
from sapien.render import RenderVRDisplay

import gymnasium as gym
import numpy as np
import h5py
import json
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
import mani_skill.trajectory.utils as trajectory_utils
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

import torch 
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_axis_angle

import time

sapien.render.set_log_level("info")
sapien.render.set_viewer_shader_dir("../vulkan_shader/vr_default")
sapien.render.enable_vr()

OPEN = 0.1
CLOSE = 0.008

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] ="BuildBlock-v1"
    """The environment ID of the task you want to simulate"""
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""
    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""
    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "bi_pd_ee_target_delta_pose"
    """Control mode"""
    render_mode: str = "rgb_array" #  "human"
    """Render mode"""
    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    record_dir: Optional[str] = './mani_skill/record'
    """Directory to save recordings"""
    save_video: Annotated[bool, tyro.conf.arg(aliases=["-v"])]  = False   # not use it
    """whether to save the videos of the demonstrations after collecting them all"""
    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-se"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
 

class VRViewer:
    def __init__(self):

        # remove this line if you use controllers and do not need hand skeleton
        # enable_hand_tracking()

        self.vr = RenderVRDisplay()
        self.controllers = self.vr.get_controller_ids()
        self.renderer_context = sapien.render.SapienRenderer()._internal_context
        self._create_visual_models()

        self.reset()

    def reset(self):
        self.controller_axes = None
        self.marker_spheres = None

        self.left_hand_spheres = None
        self.right_hand_spheres = None

    @property
    def root_pose(self):
        return self.vr.root_pose

    @root_pose.setter
    def root_pose(self, pose):
        self.vr.root_pose = pose

    @property
    def ray_angle(self):
        return np.pi / 4

    @property
    def render_scene(self):
        return self.vr._internal_scene

    @property
    def controller_poses(self):
        return [self.vr.get_controller_pose(c) for c in self.controllers]

    def set_scene(self, scene):
        self.scene = scene
        self.vr.set_scene(scene)

    def render(self):
        self._update_controller_axes()
        self.vr.update_render()
        self.vr.render()

    def pick(self, index):
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]
        c2r = self.controller_poses[index]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        d = t2w.to_transformation_matrix()[:3, 0]

        assert isinstance(self.scene.physx_system, sapien.physx.PhysxCpuSystem)
        px: sapien.physx.PhysxCpuSystem = self.scene.physx_system
        res = px.raycast(t2w.p, d, 50)
        return res

    # helper visuals
    def _create_visual_models(self):
        self.cone = self.renderer_context.create_cone_mesh(16)
        self.capsule = self.renderer_context.create_capsule_mesh(0.1, 0.5, 16, 4)
        self.cylinder = self.renderer_context.create_cylinder_mesh(16)
        self.sphere = self.renderer_context.create_uvsphere_mesh()

        self.laser = self.renderer_context.create_line_set(
            [0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]
        )

        self.mat_red = self.renderer_context.create_material(
            [5, 0, 0, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_green = self.renderer_context.create_material(
            [0, 1, 0, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_blue = self.renderer_context.create_material(
            [0, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_cyan = self.renderer_context.create_material(
            [0, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_magenta = self.renderer_context.create_material(
            [1, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_white = self.renderer_context.create_material(
            [1, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.red_cone = self.renderer_context.create_model([self.cone], [self.mat_red])
        self.green_cone = self.renderer_context.create_model(
            [self.cone], [self.mat_green]
        )
        self.blue_cone = self.renderer_context.create_model(
            [self.cone], [self.mat_blue]
        )
        self.red_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_red]
        )
        self.green_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_green]
        )
        self.blue_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_blue]
        )
        self.cyan_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_cyan]
        )
        self.magenta_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_magenta]
        )
        self.white_cylinder = self.renderer_context.create_model(
            [self.cylinder], [self.mat_white]
        )
        self.red_sphere = self.renderer_context.create_model(
            [self.sphere], [self.mat_red]
        )
        self.cyan_sphere = self.renderer_context.create_model(
            [self.sphere], [self.mat_cyan]
        )

    def _create_coordiate_axes(self):
        render_scene = self.render_scene

        node = render_scene.add_node()
        obj = render_scene.add_object(self.red_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([1, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.red_capsule, node)
        obj.set_position([0.52, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 1, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_capsule, node)
        obj.set_position([0, 0.51, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 0, 1])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_capsule, node)
        obj.set_position([0, 0, 0.5])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        # obj = render_scene.add_line_set(self.laser, node)
        # obj.set_scale([40, 0, 0])
        # obj.line_width = 20
        # ray_pose = sapien.Pose()
        # ray_pose.rpy = [0, self.ray_angle, 0]
        # obj.set_rotation(ray_pose.q)

        node.set_scale([0.025, 0.025, 0.025])

        return node

    def _update_controller_axes(self):
        if self.controller_axes is None:
            self.controller_axes = [
                self._create_coordiate_axes() for c in self.controllers
            ]

        for n, pose in zip(self.controller_axes, self.controller_poses):
            c2w = self.vr.root_pose * pose
            n.set_position(c2w.p)
            n.set_rotation(c2w.q)

    def _create_marker_sphere(self):
        node = self.render_scene.add_object(self.red_sphere)
        node.set_scale([0.05] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node

    def _create_hand_sphere(self):
        node = self.render_scene.add_object(self.cyan_sphere)
        node.set_scale([0.01] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node

    def update_hand_skeleton(self):
        root_pose = self.root_pose
        hrp = self.vr.get_left_hand_root_pose()
        poses = self.vr.get_left_hand_skeletal_poses()
        if len(poses) != 31:
            return

        if self.left_hand_spheres is None:
            self.left_hand_spheres = [self._create_hand_sphere() for _ in range(25)]

        for s, p in zip(self.left_hand_spheres, poses[1:]):
            sphere_pose = root_pose * hrp * p
            s.transparency = 0
            s.set_position(sphere_pose.p)

        hrp = self.vr.get_right_hand_root_pose()
        poses = self.vr.get_right_hand_skeletal_poses()
        if len(poses) != 31:
            return

        if self.right_hand_spheres is None:
            self.right_hand_spheres = [self._create_hand_sphere() for _ in range(25)]

        for s, p in zip(self.right_hand_spheres, poses[1:]):
            sphere_pose = root_pose * hrp * p
            s.transparency = 0
            s.set_position(sphere_pose.p)

    def update_ee_pose(self):
        root_pose = self.root_pose
        left_controller_pose = self.controller_poses[0] #self.vr.get_controller_pose(self.controllers[0])

        if self.left_hand_spheres is None:
            self.left_hand_spheres = self._create_hand_sphere() 
            # self.left_hand_spheres_extend = self._create_hand_sphere()   # add for debug
        
        left_abs_pose = root_pose * left_controller_pose 
        self.left_hand_spheres.transparency = 0
        self.left_hand_spheres.set_position(left_abs_pose.p)
        # print("left_abs_pose", left_abs_pose)

        # self.left_hand_spheres_extend.transparency = 0
        # self.left_hand_spheres_extend.set_position(left_abs_pose.p + np.array([0, 0.05, 0]))   #add for debug

        right_controller_pose = self.controller_poses[1] #self.vr.get_controller_pose(self.controllers[1])

        if self.right_hand_spheres is None:
            self.right_hand_spheres = self._create_hand_sphere() 
        
        right_abs_pose = root_pose * right_controller_pose 
        self.right_hand_spheres.transparency = 0
        self.right_hand_spheres.set_position(right_abs_pose.p)
        # print("right_abs_pose", right_abs_pose)

    def update_marker_sphere(self, i, hit):
        if self.marker_spheres is None:
            self.marker_spheres = [
                self._create_marker_sphere() for c in self.controllers
            ]

        if hit is None:
            self.marker_spheres[i].transparency = 1
        else:
            self.marker_spheres[i].set_position(hit.position)
            self.marker_spheres[i].transparency = 0


def run(env):
    # scene = sapien.Scene()
    # from sapien_demo_arena import DemoArena
    # DemoArena().load(scene)
    scene = env.scene.sub_scenes[0]

    viewer = VRViewer()
    viewer.set_scene(scene)

    viewer.root_pose = sapien.Pose([-1, -0.3 , -0.65], [-1, 0, 0, 0]) # 001 0001  人的视角的位置

    right_pressed = OPEN
    left_pressed = OPEN
    pre_lp = None
    pre_rp = None
    # prev_button_pressed = [0 for c in viewer.controllers]
    while True:
        # scene.step() #
        viewer.render()

        for i, c in enumerate(viewer.controllers):
            button_pressed = viewer.vr.get_controller_button_pressed(c)
            # changed = button_pressed ^ prev_button_pressed[i]

            # trigger down   原本注释了
            # if changed & 0x200000000 and button_pressed & 0x200000000:
            #     viewer.pick(i)

            # continuously test
            # print("button_pressed", button_pressed)
            if button_pressed & 0x200000000:
                if i == 0:
                    left_pressed = CLOSE #
                elif i == 1:
                    right_pressed = CLOSE #
                else:
                    print("irrelated controller button pressed")
                hit = viewer.pick(i)
                viewer.update_marker_sphere(i, hit)
            else:
                if i == 0:
                    left_pressed = OPEN #
                elif i == 1:
                    right_pressed = OPEN #
                else:
                    print("irrelated controller button pressed")
                viewer.update_marker_sphere(i, None)

            # viewer.update_hand_skeleton()
            viewer.update_ee_pose()
            # prev_button_pressed[i] = button_pressed
        
        
        cur_lp = viewer.controller_poses[0]
        cur_rp = viewer.controller_poses[1]
        if pre_lp is None:
            pre_lp = cur_lp
        if pre_rp is None:
            pre_rp = cur_rp
        lp = pre_lp.inv() * cur_lp  # 现在是 target delta pose control
        rp = pre_rp.inv() * cur_rp
        # 加入一些手臂之间的展宽，旋转不要发生改变; y方向上，左手方向是y正方向
        lp.p[1] = lp.p[1] + 0.05 
        rp.p[1] = rp.p[1] - 0.05

        lg = [left_pressed]  # 目前给的是开合的数值（不是相对值？）
        rg = [right_pressed]
        print("------------------------------------------")
        action = np.hstack([lp.p, quaternion_to_axis_angle(torch.tensor(lp.q)), lg, rp.p, quaternion_to_axis_angle(torch.tensor(rp.q)), rg])  
        obs, reward, terminated, truncated, info = env.step(action)  # 包含scene.step()
        # print("action", action)
        # print("reward", reward)
        # print("terminated", terminated)

        pre_lp = cur_lp
        pre_rp = cur_rp

        # time.sleep(0.1)
        
        v = env.render_human()
        
        if v.window.key_press("q"):
            return "q"
        elif v.window.key_press("c"):
            return "c"
        elif v.window.key_press("r"):
            return "r"


def main(args: Args):
    try:
        output_dir = f"{args.record_dir}/{args.env_id}/env/{args.seed}/test1"
        np.set_printoptions(suppress=True, precision=3)
        if isinstance(args.seed, int):
            init_seed = args.seed # 保存
            args.seed = [args.seed]
            print("args.seed", args.seed)
        if args.seed is not None:  ##???
            np.random.seed(args.seed[0])
        parallel_in_single_scene = args.render_mode == "human"
        if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
            print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
            parallel_in_single_scene = False
        if args.render_mode == "human" and args.num_envs == 1:
            parallel_in_single_scene = False
        print("---- obs:", args.obs_mode)
        env_kwargs = dict(
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            reward_mode="none",  #"sparse"
            enable_shadow=True,
            parallel_in_single_scene=parallel_in_single_scene,
        )
        if args.robot_uids is not None:
            env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        env: BaseEnv = gym.make(
            args.env_id,
            **env_kwargs
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video= True, 
            # video_fps=10,
            info_on_video=False,
            source_type="teleoperation",
            source_desc="teleoperation via the vr system",
            # max_steps_per_video=gym_utils.find_max_episode_steps_value(env)
        )

        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        
        num_trajs = 0
        seed_val=init_seed
        seed = [seed_val]
        env.reset(seed=seed)
        
        while True:
            print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
            key = run(env)
            if key == "q":
                num_trajs += 1
                break
            elif key == "c":
                seed_val += 1
                seed = [seed_val]
                num_trajs += 1
                print("---------- continue----- Resetting env with seed:", seed_val)
                env.reset(seed=seed)
                continue
            elif key == "r":
                num_trajs += 1
                env.reset(seed=seed, options=dict(save_trajectory=False))  


        #  其实 self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")
        h5_file_path = env._h5_file.filename
        json_file_path = env._json_path
        print("h5_file_path:", h5_file_path)
        print("json_file_path:", json_file_path)
        env.close()
        del env

    except KeyboardInterrupt:
        pass
        


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)