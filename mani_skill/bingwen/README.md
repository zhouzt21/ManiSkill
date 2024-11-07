## Config
1. **create your own conda environment**

``` bash
conda create -n maniskill python=3.9
git clone https://github.com/haosulab/ManiSkill.git  # the old version dont include robocasa
cd ManiSkill && pip install -e .
pip install torch torchvision torchaudio
```
If you have problems about type checking error. you can remove the pyproject.toml and pyrightconfig.json

2. **create your own environment**:
- you can copy mani_skill/envs/tasks/mobile_manipulation/robocasa/kitchen.py to a new file,eg,mani_skill/envs/tasks/mobile_manipulation/robocasa/custom_kitchen.py. 
- you should rename the env_id in the @register_env() function.
- you should modify the __init__.py in the  directory. such as add `from .custom_kitchen import RoboCasaCustomKitchenEnv`.


3. **Test your work**:
``` bash
  python -m mani_skill.examples.demo_random_action \
  -e "RoboCasaCustomKitchen" \
  -n 4 -s 0 1 2 3 \
  --render-mode="human" -r "fetch"
```

<!-- 
**To add the object in the robocasa environment you should:**
1. create a robocasa environment.
2. use current conda environment and cd robocasa directory to use `pip install -e .`.
3. you can use the code `new_base_path = os.path.join(robocasa.models.assets_root, "objects")`.
4. you should `import os` and `import robocasa` at the top of the environment file.
` -->


## Others

You can use hard code now. But you can use yaml to make the scene easier.
But you should learn yaml yourself.


**robocasa important directories**:
- mani_skill/utils/scene_builder/robocasa/objects/kitchen_objects.py
- mani_skill/utils/scene_builder/robocasa/

**you can check this to find whether robocasa is better than replicacad.**
- mani_skill/utils/scene_builder/replicacad/scene_builder.py
- ~/.maniskill/data/scene_datasets/replica_cad_dataset

**fixtures_name:**
- wall_room
- wall_backing_room
- wall_left_room
- wall_left_backing_room
- wall_right_room
- wall_right_backing_room
- outlet_room
- outlet_2_room
- light_switch_room
- light_switch_2_room
- utensil_rack_room
- floor_room
- floor_backing_room
- counter_corner_main_group
- counter_main_main_group
- counter_corner_2_main_group
- cab_corner_main_group
- cab_corner_2_main_group
- cab_main_main_group
- window_group_main_group
- cab_1_main_group
- cab_corner_3_main_group
- cab_corner_4_main_group
- stack_1_main_group_base
- stack_1_main_group_1
- stack_1_main_group_2
- stack_1_main_group_3
- stack_2_main_group_base
- stack_2_main_group_1
- stack_2_main_group_2
- stack_2_main_group_3
- stack_3_main_group_base
- stack_3_main_group_1
- stack_3_main_group_2
- stack_3_main_group_3
- stack_3_main_group_4
- toaster_main_group
- knife_block_main_group
- counter_1_front_group
- counter_corner_front_group
- plant_front_group
- stool_1_front_group
- stool_2_front_group
- counter_1_left_group
- stove_left_group
- counter_2_left_group
- counter_corner_left_group
- fridge_left_group
- fridge_housing_left_group
- stack_1_left_group_base
- stack_1_left_group_1
- stack_1_left_group_2
- stack_2_left_group_base
- stack_2_left_group_1
- stack_2_left_group_2
- cab_1_left_group
- microwave_left_group
- cab_micro_left_group
- cab_2_left_group
- cab_3_left_group
- coffee_machine_left_group
- utensil_holder_left_group
- sink_right_group
- counter_1_right_group
- counter_corner_right_group
- counter_corner_2_right_group
- stack_1_right_group_base
- stack_1_right_group_1
- stack_1_right_group_2
- dishwasher_right_group
- stack_2_right_group_base
- stack_2_right_group_1
- stack_2_right_group_2
- stack_2_right_group_3
- stack_2_right_group_4
- cab_1_right_group
- shelves_right_group
- paper_towel_right_group

[NOTE]： if you use reconfigure as True in `obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=False))`, you may encounter the false in the get_fixture() functions becasue it have the different fixtures names.
