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

# Here modify the max_steps
@register_env(
    "RoboCasaCustomKitchen-v1", max_episode_steps=1000, asset_download_ids=["RoboCasa"]
)
class RoboCasaCustomKitchenEnv(RoboCasaKitchenEnv):
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
        layout_ids=None,
        style_ids=None,
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
        # TODO To be changed.
        return "put the apple in the bowl"

    # copy from robocasa, can not be used in maniskill
    def get_ep_meta(self):
        """
        Returns a dictionary containing episode meta data
        """

        def copy_dict_for_json(orig_dict):
            new_dict = {}
            for (k, v) in orig_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = copy_dict_for_json(v)
                elif isinstance(v, Fixture):
                    new_dict[k] = v.name
                else:
                    new_dict[k] = v
            return new_dict

        ep_meta = super().get_ep_meta()
        ep_meta["layout_id"] = self.layout_id
        ep_meta["style_id"] = self.style_id
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]
        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""

        ep_meta["fixture_refs"] = dict(
            {k: v.name for (k, v) in self.fixture_refs.items()}
        )
        ep_meta["cam_configs"] = deepcopy(self._cam_configs)

        return ep_meta


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
        Cfgs:

            info(dict): include the "mjcf_path"

                mjcf_path(str): the path of obj/model.xml

            name(str): define the name of object,
            
            type(str): "object" or "fxiture"(not use in object)
            
            obj_groups(list or str): groups to sample from or the exact xml path of the object to spawn -> helps you to find the xml.

            exclude_obj_groups(str or list): groups to exclude  eg: exclude_obj_groups = ["plate", "pan", "vegetable"]
            
            [optional]graspable (bool): whether the sampled object must be graspable

            [optional]washable (bool): whether the sampled object must be washable

            [optional]microwavable (bool): whether the sampled object must be microwavable

            [optional]cookable (bool): whether whether the sampled object must be cookable

            [optional]freezable (bool): whether whether the sampled object must be freezable

            max_size(tuple with 3 elements): max_size is used to specify the maximum size limits of an object in three dimensions,
                ensuring that the object can fit into a specific placement area or container in simulation or robotic operations.
                By setting max_size, it helps prevent placement failure or collision issues caused by the object being too large.
                eg: max_size=(0.35, 0.45, None), or =(None, None, 0.10),
                max_size = (x_max, y_max, z_max), max_size for object not the ref. None represents there is no limit in
                this dimension.


            placement: 

                fixture(str or ClassType): find in the scene_data["fixtures"], ref get_fixtures(): 
                    fixture can be a class or an id string. If not using ref,
                    the pos is base on fixture's pos.
                    [Note] fixture should in the scene, otherwise it will return error.
                                            
                sample_region_kwargs(dict optional): When we want to place an object at a specific location,there are usually some positional constraints, 
                    meaning the object cannot be placed anywhere, but must be placed within a specific area.
                    This area is called the sampling region. The sampling region is usually determined based on the position, size, and other scene conditions of the reference object.
                
                    ref (str or ClassType): reference fixture(like fixture above) used in determining sampling location.

                    loc (str): sampling method, one of ["nn", "left", "right", "left_right", "any"]
                        nn: chooses the closest top geom to the reference fixture
                        left: chooses the any top geom within 0.3 distance of the left side of the reference fixture
                        right: chooses the any top geom within 0.3 distance of the right side of the reference fixture
                        left_right: chooses the any top geom within 0.3 distance of the left or right side of the reference fixture, random choose or meet some condition
                        any: chooses any top geom

                    top_size (tuple): minimum size of the top region to return
                            make sure sampled counter region is large enough to place the object


                size(tuple with 2 elements): (width, hegiht):This indicates the minimum size requirement for the placement area is (0.2, 0.2), 
                    meaning that the width and height of the area must be at least 0.2. It doesn't mean the object is that size.

                pos(tuple with 2 elements): (x_m, y_m) the distance away from the ref. If 'ref' is not set, the pos is relative to the fixture.
                    the first element can be 'ref',representing the x-distance relative to 'ref', and pos[1] represents the offset in the y-direction. 

                offset(tuple with 2 elements): (x_m,y_m) on the basis of pos. It add more flexibility.
                    if you want to offset the x-coordinate relative to 'ref', you need to add an offset and set 'ref' on the x-axis.

                rotation(list/tuple with 2 elements): eg: rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    eg: rotation=(2 * np.pi / 8, 3 * np.pi / 8); rotation=np.pi / 2,
                
                ensure_object_boundary_in_range(bool): ? usually False

                ensure_object_in_region(bool): ?

                ensure_valid_placement(bool): ?

                try_to_place_in(str eg: "tray","container","cutting_board"): ?


                [Note]: Both pos and offset can be set, but it seems that offset provides more flexibility in placing the object.
                    Sometimes, the placed position may not match the actual position, which might be related to collisions.
                    Z is excluded for it is height.

        Returns:
            Cfgs: give object info to env to create objects.
        """

        # obj_model_path = os.path.join(
        #     ROBOCASA_OBJAVERSE_DIR, "apple/apple_0/model.xml"
        # )
        # for index in range(self.num_envs):
        #     cfgs.append(
        #         dict(
        #             info = {"mjcf_path": obj_model_path,},
        #             type = "object",
        #             name = "obj_apple_0",
        #             obj_groups = None,
        #             placement = dict(
        #                 fixture=self.fixture_refs[index]["counter"],
        #                 sample_region_kwargs=dict(
        #                     ref=self.fixture_refs[index]["sink"],
        #                 ),
        #                 size=(0.2, 0.2),
        #                 pos=('ref',-0.2),
        #                 offset=(0.5, 0),
        #                 rotation=(0,0)
        #                 ),
        #         ) 
        #     )

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "apple/apple_0/model.xml"
        )
        cfgs.append(
            dict(
                info = {"mjcf_path": obj_model_path,},
                type = "object",
                name = "obj_apple_0",
                obj_groups = None,
                placement = dict(
                        fixture='counter_main_main_group',
                        sample_region_kwargs=dict(
                            ref='knife_block_main_group',
                        ),
                        size=(0.2, 0.2),
                        pos=('ref',-0.2),
                        offset=(0.5, 0),
                        rotation=(0,0)
                        ),
                ) 
            )

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "bowl/bowl_1/model.xml"
        )
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = "object",
                name = "obj_bowl",
                obj_groups = None,
                placement = dict(
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.5, 0.0),
                        rotation=(0,0)
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "apple/apple_10/model.xml"
        )
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = "object",
                name = "obj_apple_10",
                obj_groups = None,
                placement = dict(
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.8, 0.05),
                        rotation=(0,0)
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "banana/banana_1/model.xml"
        )
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = "object",
                name = "obj_banana_1",
                obj_groups = None,
                placement = dict(
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(1.0, -0.03),
                        rotation=(0,0) 
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "carrot/carrot_1/model.xml"
        )
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = "object",
                name = "obj_carrot_1",
                obj_groups = None,
                placement = dict(
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.6, -0.034),
                        rotation=(-0.1,0.1)
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "milk/milk_1/model.xml"
        )
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = "object",
                name = "obj_milk_1",
                obj_groups = None,
                placement = dict(
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='counter_1_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.9, 0.04),
                        rotation=[-0.1,0.1] 
                        ),
                ) 
            )

        return cfgs