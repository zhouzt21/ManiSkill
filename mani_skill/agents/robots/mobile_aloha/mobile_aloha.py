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


# FIXME: EE control is not ready!!!


@register_agent()
class MobileAloha(BaseAgent):
    uid = "mobile_aloha"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/mobile_aloha/urdf/aloha_sapien_sim.urdf"
    srdf_path = f"{PACKAGE_ASSET_DIR}/robots/mobile_aloha/srdf/aloha_sapien_sim.srdf"

    @property
    def _sensor_configs(self):
        return [
            # CameraConfig(
            #     uid="mobile_aloha_base",
            #     pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
            #     width=128,
            #     height=128,
            #     fov=2,
            #     near=0.01,
            #     far=100,
            #     entity_uid="camera_base_link",
            # ),
            CameraConfig(
                uid="mobile_aloha_base_low_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="camera_link1",
            ),
            CameraConfig(
                uid="mobile_aloha_base_high_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="camera_link2",
            ),
            CameraConfig(
                uid="mobile_aloha_left_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="left_camera",
            ),
            CameraConfig(
                uid="mobile_aloha_right_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
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
        self.fl_ee_link_name = "fl_link6"
        self.fr_ee_link_name = "fr_link6"
        self.fl_gripper_joint_names = ["fl_joint7", "fl_joint8"] # front-left
        self.fr_gripper_joint_names = ["fr_joint7", "fr_joint8"] # front-right

        self.arm_base_link_name = self.fl_arm_base_link_name
        self.arm_joint_names = self.fl_arm_joint_names
        self.ee_link_name = self.fl_ee_link_name
        self.gripper_joint_names = self.fl_gripper_joint_names

        # FIXME: wheels

        self.joint_stiffness = 1000
        self.joint_damping = 200

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
            normalize_action=False
        )
        arm_config_fn_dict["pd_joint_delta_pos"] = partial(
            PDJointPosControllerConfig,
            lower=-0.1, upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            use_delta=True,
        )
        arm_config_fn_dict["pd_joint_target_delta_pos"] = partial(
            PDJointPosControllerConfig,
            lower=-0.1, upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            use_delta=True,
            use_target=True,
        )

        # PD EE position
        arm_config_fn_dict["pd_ee_delta_pos"] = partial(
            PDEEPosControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
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
        )
        arm_config_fn_dict["pd_ee_target_delta_pos"] = partial(
            PDEEPosControllerConfig,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            urdf_path=self.urdf_path,
            use_target=True,
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
            frame="ee_align"
        )

        # PD joint velocity
        arm_config_fn_dict["pd_joint_vel"] = partial(
            PDJointVelControllerConfig,
            lower=-1.0,
            upper=1.0,
            damping=self.joint_damping,  # this might need to be tuned separately
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

            controller_configs[control_mode] = dict(
                arm=config_fn(**kwargs1),
                gripper=gripper_pd_joint_pos_fn(
                    joint_names=self.fl_gripper_joint_names
                )
            )

            controller_configs["bi_" + control_mode] = dict(
                arm1=config_fn(**kwargs1),
                gripper1=gripper_pd_joint_pos_fn(
                    joint_names=self.fl_gripper_joint_names
                ),
                arm2=config_fn(**kwargs2),
                gripper2=gripper_pd_joint_pos_fn(
                    joint_names=self.fr_gripper_joint_names
                )
            )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.
        """
        qpos = []
        qvel = []
        for ctrl_name, ctrl in self.controller.controllers.items():
            if 'arm' in ctrl_name:
                qpos.append(ctrl.qpos)
                qvel.append(ctrl.qvel)
            elif 'gripper' in ctrl_name:
                qpos.append(ctrl.qpos[..., :1])
                qvel.append(ctrl.qvel[..., :1])
            else:
                import pdb; pdb.set_trace()

        qpos = torch.concat(qpos, dim=-1)
        qvel = torch.concat(qvel, dim=-1)
        obs = dict(qpos=qpos, qvel=qvel)

        return obs

    def _after_init(self):
        return super()._after_init()
    
    def get_arm_base_pose(self, arm_id: int = 0):
        assert arm_id in [0, 1]
