from copy import deepcopy
from typing import Dict

import numpy as np
import sapien
import torch
import os

from mani_skill.utils.scene_builder.robocasa.scene_builder import FIXTURES,FIXTURES_INTERIOR
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType, Fixture

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
    sample_kitchen_object,
)
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.robocasa.utils import scene_registry
from mani_skill.utils.scene_builder.robocasa.utils.placement_samplers import (
    RandomizationError,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR

from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv

ROBOCASA_OBJAVERSE_DIR = ROBOCASA_ASSET_DIR / "objects/objaverse"

@register_env(
    "RoboCasaStackCan-v1", max_episode_steps=1000, asset_download_ids=["RoboCasa"]
)
class RoboCasaStackCanEnv(RoboCasaKitchenEnv):
    SUPPORTED_ROBOTS = ["fetch", "none", "mobile_aloha"]
    SUPPORTED_REWARD_MODES = ["none"]
    EXCLUDE_LAYOUTS = []

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,  # currently unused variable
        reward_scale=1.0,  # currently unused variable
        reward_shaping=False,  # currently unused variables
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=25,
        horizon=1000,
        ignore_done=False,
        # hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mujoco",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=[6],
        style_ids=[0],
        scene_split=None,  # unsued, for backwards compatibility
        generative_textures=None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
        fixtures_only=False,
        **kwargs,
    ):        
        all_kwargs = locals()
        all_kwargs.pop("self")
        all_kwargs.pop("args")
        all_kwargs.pop("kwargs")
        all_kwargs.pop("__class__")
        all_kwargs.update(kwargs)
        super().__init__(*args, **all_kwargs)

    def get_task_description(self):
        return "Pick up the cans one by one and stack them together."
    
    # can be used for create objects.
    def _setup_kitchen_references(self):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        # serialized_refs = self._ep_meta.get("fixture_refs", {})
        serialized_refs = {
            "counter" : {
                "id" : FixtureType.COUNTER, 
                "ref" : None, 
                "size" : (1.0, 0.4)
            },

            "sink" : {
                "id" : FixtureType.SINK, 
                "ref" : None, 
                "size" : (0.8, 0.5)
            },
        }
        self.fixture_refs[self._scene_idx_to_be_loaded] = {
            k: self.scene_builder.get_fixture(
                self.scene_builder.scene_data[self._scene_idx_to_be_loaded]["fixtures"],
                **v,
            )
            for (k, v) in serialized_refs.items()
        }

    def _get_obj_cfgs(self):
        cfgs = []

        """
        ALL_CAN = can_0  can_11  can_12  can_13  can_15  can_16  can_17  can_18  can_2  can_3  can_4  can_6  can_8  can_9
        """

        can_variations = [
            # "can_0",
            # "can_11",
            # "can_12",
            # "can_13",
            # "can_15",
            # "can_16",
            # "can_17",
            # "can_18",
            # "can_2",
            # "can_3",
            # "can_4",
            # "can_6",
            # "can_8",
            # "can_9",
            "can_8", "can_8", "can_8"
        ]

        for i in range(len(can_variations)):
            obj_model_path = os.path.join(
                ROBOCASA_OBJAVERSE_DIR, f"can/{can_variations[i]}/model.xml")
            
            cfgs.append(
                dict(
                    info = {"mjcf_path": obj_model_path},
                    type = "object",
                    name = f"obj_can_{i}",
                    obj_groups = None,
                    placement = dict(
                        fixture='counter_main_main_group',
                        sample_region_kwargs=dict(
                            ref='utensil_holder_main_group',
                        ),
                        size=(0.2, 0.2),
                        pos=('ref', -0.2),
                        offset=(np.random.rand() * -0.5 -0.5, np.random.rand() * -0.3),
                        rotation=(0,0)
                    )
                )
            )

        return cfgs
