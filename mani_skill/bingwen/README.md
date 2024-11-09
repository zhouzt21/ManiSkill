## Config
1. **create your own conda environment**

``` bash
conda create -n maniskill python=3.10 && conda activate maniskill
git clone https://github.com/haosulab/ManiSkill.git  # the old version of Maniskill doesn't include robocasa
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


## Others

You can use hard code now. But you can use yaml to make the scene easier.
But you should learn yaml yourself.


**Directories**:
- mani_skill/envs/tasks/mobile_manipulation/robocasa/kitchen.py
- mani_skill/utils/scene_builder/robocasa/scene_builder.py
- mani_skill/utils/scene_builder/robocasa/objects/kitchen_objects.py

**you can check this to find whether robocasa is better than replicacad.**
- mani_skill/utils/scene_builder/replicacad/scene_builder.py
- ~/.maniskill/data/scene_datasets/replica_cad_dataset


[NOTE]： if you use reconfigure as True in `obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=False))`, you may encounter the false in the get_fixture() functions becasue it have the different fixtures names.

**Put your own objects into the scene**
```python 
    def _get_obj_cfgs(self):
        cfgs = []
        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "apple/apple_0/model.xml"
        )
        # The name should in the scene, otherwise it will return error.
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_apple_0",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_main_main_group',
                        sample_region_kwargs=dict(
                            ref='knife_block_main_group',
                        ),
                        size=(0.2, 0.2),
                        pos=('ref',-0.2),# x y exclude z(height)
                        offset=(0.5, 0),# x y exclude z(height)
                        rotation=(0,0)
                        ),
                ) 
            )
        # 'ref' 只能作为pos[0]，代表相对于ref而言，pos[1]代表偏移的y的距离，此时若不加offset，则x与'ref'相同
        # 此时如果想要在ref的基础上向x偏移，则需要加入offset，在x上设置即可。
        # pos 和 offset的区别？ pos和offset均可设置，但是好像offset会把物体放的更灵活一点，有的时候放置位置和实际位置不符。可能与碰撞有关。
        # ref可以不设置，这样的pos就是相对于fixture的。
        return cfgs
```