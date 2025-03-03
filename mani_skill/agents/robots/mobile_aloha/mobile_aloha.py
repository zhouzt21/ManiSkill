from copy import deepcopy
from functools import partial
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array


@register_agent()
class MobileAloha(BaseAgent):
    uid = "mobile_aloha"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/mobile_aloha/urdf/aloha_sapien_sim.urdf"
    srdf_path = f"{PACKAGE_ASSET_DIR}/robots/mobile_aloha/srdf/aloha_sapien_sim.srdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="cam_high",
                pose=Pose.create_from_pq([0, 0, 0.15], [1, 0, 0, 0]),
                width=640,
                height=480,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="camera_link1",
            ),
            CameraConfig(
                uid="cam_top",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=640,
                height=480,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="camera_link2",
            ),
            CameraConfig(
                uid="cam_left_wrist",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=640,
                height=480,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="left_camera",
            ),
            CameraConfig(
                uid="cam_right_wrist",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=640,
                height=480,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="right_camera",
            ),
        ]
    
    def __init__(self, *args, **kwargs):
        # arms
        self.fl_arm_base_link_name = "fl_base_link"
        self.fr_arm_base_link_name = "fr_base_link"
        self.fl_arm_joint_names = [f"fl_joint{i+1}" for i in range(6)] # front-left (main)
        self.fr_arm_joint_names = [f"fr_joint{i+1}" for i in range(6)] # front-right
        self.lr_arm_joint_names = [f"lr_joint{i+1}" for i in range(6)] # left-rear
        self.rr_arm_joint_names = [f"rr_joint{i+1}" for i in range(6)] # left-rear
        # self.arm_force_limit?

        # grippers
        self.fl_ee_link_name = "fl_end_effector"
        self.fr_ee_link_name = "fr_end_effector"
        self.fl_gripper_joint_names = ["fl_joint7", "fl_joint8"] # front-left
        self.fr_gripper_joint_names = ["fr_joint7", "fr_joint8"] # front-right

        self.arm_base_link_name = self.fl_arm_base_link_name
        self.arm_joint_names = self.fl_arm_joint_names
        self.ee_link_name = self.fl_ee_link_name
        self.gripper_joint_names = self.fl_gripper_joint_names

        # FIXME: wheels

        self.joint_stiffness = 1000
        self.joint_damping = 200
        self.arm_force_limit = 100
        self.arm_joint_limits = None

        super().__init__(*args, **kwargs)


    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_config_fn_dict = dict()

        arm_config_fn_dict["pd_joint_pos"] = partial(
            PDJointPosControllerConfig,
            lower=None, upper=None,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            normalize_action=False, #True,
        )
        arm_config_fn_dict["pd_joint_delta_pos"] = partial(
            PDJointPosControllerConfig,
            lower=-0.1, upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            use_delta=True,
            normalize_action=False, #True,
        )
        arm_config_fn_dict["pd_joint_target_delta_pos"] = partial(
            PDJointPosControllerConfig,
            lower=-0.1, upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            use_delta=True,
            use_target=True,
            normalize_action=False, #True,
        )

        # PD EE position
        arm_config_fn_dict["pd_ee_delta_pos"] = partial(
            PDEEPosControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
            normalize_action=False, #True,
        )
        arm_config_fn_dict["pd_ee_delta_pose"] = partial(
            PDEEPoseControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
            normalize_action=False, #True,
        )
        arm_config_fn_dict["pd_ee_target_delta_pos"] = partial(
            PDEEPosControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
            use_target=True,
            normalize_action=False, #True,
        )
        arm_config_fn_dict["pd_ee_target_delta_pose"] = partial(
            PDEEPoseControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
            use_target=True,
            normalize_action=False, #True,
        )

        # PD ee position (for human-interaction/teleoperation)
        arm_config_fn_dict["pd_ee_delta_pose_align"] = partial(
            PDEEPoseControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
            frame="ee_align",
            normalize_action=False, #True,
        )

        # PD joint velocity
        arm_config_fn_dict["pd_joint_vel"] = partial(
            PDJointVelControllerConfig,
            lower=-1.0,
            upper=1.0,
            damping=self.joint_damping,  # this might need to be tuned separately
            normalize_action=False, #True,
        )

        # PD joint position and velocity
        arm_config_fn_dict["pd_joint_pos_vel"] = partial(
            PDJointPosVelControllerConfig,
            lower=None,
            upper=None,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            normalize_action=False, #True,
        )
        arm_config_fn_dict["pd_joint_delta_pos_vel"] = partial(
            PDJointPosVelControllerConfig,
            lower=-0.1,
            upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            use_delta=True,
            normalize_action=False, #True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos_fn = partial(
            PDJointPosMimicControllerConfig,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.045,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            normalize_action=False, #True,
        )

        controller_configs = dict()

        for control_mode, config_fn in arm_config_fn_dict.items():
            if "ee" in control_mode:
                kwargs1 = dict(joint_names=self.fl_arm_joint_names,
                               ee_link=self.fl_ee_link_name)
                kwargs2 = dict(joint_names=self.fr_arm_joint_names,
                               ee_link=self.fr_ee_link_name)
            else:
                kwargs1 = dict(joint_names=self.fl_arm_joint_names)
                kwargs2 = dict(joint_names=self.fr_arm_joint_names)

            # two mini modes for each  mode
            controller_configs[control_mode] = dict(
                arm=config_fn(**kwargs1),
                gripper=gripper_pd_joint_pos_fn(
                    joint_names=self.fl_gripper_joint_names
                )
            )
            # two mini modes for each  mode
            controller_configs["bi_" + control_mode] = dict(
                arm_left=config_fn(**kwargs1),
                gripper_left=gripper_pd_joint_pos_fn(
                    joint_names=self.fl_gripper_joint_names
                ),
                arm_right=config_fn(**kwargs2),
                gripper_right=gripper_pd_joint_pos_fn(
                    joint_names=self.fr_gripper_joint_names
                )
            )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.
        """
        if 'bi' in self.control_mode:
            qpos_l = self.controller.controllers['arm_left'].qpos
            qvel_l = self.controller.controllers['arm_left'].qvel
            if 'gripper_left' in self.controller.controllers:
                qpos_gripper_l = self.controller.controllers['gripper_left'].qpos[..., :1]
                qvel_gripper_l = self.controller.controllers['gripper_left'].qvel[..., :1]

                qpos_l = torch.concat([qpos_l, qpos_gripper_l], dim=-1)
                qvel_l = torch.concat([qvel_l, qvel_gripper_l], dim=-1)
            
            qpos_r = self.controller.controllers['arm_right'].qpos
            qvel_r = self.controller.controllers['arm_right'].qvel
            if 'gripper_right' in self.controller.controllers:
                qpos_gripper_r = self.controller.controllers['gripper_right'].qpos[..., :1]
                qvel_gripper_r = self.controller.controllers['gripper_right'].qvel[..., :1]

                qpos_r = torch.concat([qpos_r, qpos_gripper_r], dim=-1)
                qvel_r = torch.concat([qvel_r, qvel_gripper_r], dim=-1)

            obs = dict(qpos_l=qpos_l, qvel_l=qvel_l, qpos_r=qpos_r, qvel_r=qvel_r)

        else:
            qpos = self.controller.controllers['arm'].qpos
            qvel = self.controller.controllers['arm'].qvel
            if 'gripper' in self.controller.controllers:
                qpos_gripper = self.controller.controllers['gripper'].qpos[..., :1]
                qvel_gripper = self.controller.controllers['gripper'].qvel[..., :1]

                qpos = torch.concat([qpos, qpos_gripper], dim=-1)
                qvel = torch.concat([qvel, qvel_gripper], dim=-1)

            obs = dict(qpos=qpos, qvel=qvel)

        return obs

    def _after_init(self):
        return super()._after_init()
    
    def get_state(self) -> Dict:
        """Get current state, including robot state and controller state"""
        state = dict()
        
        # robot state
        state["robot_root_pose"] = self.robot.root.pose
        state["robot_root_vel"] = self.robot.root.get_linear_velocity()
        state["robot_root_qvel"] = self.robot.root.get_angular_velocity()

        fl_arm_joints = [self.robot.find_joint_by_name(joint_name) for joint_name in self.fl_arm_joint_names]
        fr_arm_joints = [self.robot.find_joint_by_name(joint_name) for joint_name in self.fr_arm_joint_names]
        fl_gripper_joints = [self.robot.find_joint_by_name(joint_name) for joint_name in self.fl_gripper_joint_names]
        fr_gripper_joints = [self.robot.find_joint_by_name(joint_name) for joint_name in self.fr_gripper_joint_names]

        state["fl_arm_qpos"] = torch.stack([joint.qpos for joint in fl_arm_joints], dim=-1)
        state["fr_arm_qpos"] = torch.stack([joint.qpos for joint in fr_arm_joints], dim=-1)
        state["fl_gripper_qpos"] = torch.stack([joint.qpos for joint in fl_gripper_joints], dim=-1)
        state["fr_gripper_qpos"] = torch.stack([joint.qpos for joint in fr_gripper_joints], dim=-1)

        state["fl_arm_qvel"] = torch.stack([joint.qvel for joint in fl_arm_joints], dim=-1)
        state["fr_arm_qvel"] = torch.stack([joint.qvel for joint in fr_arm_joints], dim=-1)
        state["fl_gripper_qvel"] = torch.stack([joint.qvel for joint in fl_gripper_joints], dim=-1)
        state["fr_gripper_qvel"] = torch.stack([joint.qvel for joint in fr_gripper_joints], dim=-1)

        state["fl_EE_pose"] = self.robot.find_link_by_name(self.fl_ee_link_name).pose.raw_pose
        state["fr_EE_pose"] = self.robot.find_link_by_name(self.fr_ee_link_name).pose.raw_pose

        # controller state
        state["controller"] = self.controller.get_state()

        return state

    def is_grasping(self, object: Actor | None = None):
        return False

    def is_static(self, threshold: float):

        return super().is_static(threshold)