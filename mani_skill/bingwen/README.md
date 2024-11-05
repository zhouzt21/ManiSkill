## Config
**create your own conda environment**

``` bash
conda create -n maniskill python=3.9
git clone https://github.com/haosulab/ManiSkill.git  # the old version dont include robocasa
cd ManiSkill && pip install -e .
pip install torch torchvision torchaudio
```
If you have problems about type checking error. you can remove the pyproject.toml and pyrightconfig.json

**create your own environment**:
- you can copy mani_skill/envs/tasks/mobile_manipulation/robocasa/kitchen.py to a new file,eg,mani_skill/envs/tasks/mobile_manipulation/robocasa/custom_kitchen.py. 
- you should rename the env_id in the @register_env() function.
- you should modify the __init__.py in the  directory. such as add `from .custom_kitchen import RoboCasaCustomKitchenEnv`.


Then, you can use the command below in the bash:

``` bash
  python -m mani_skill.examples.demo_random_action \
  -e "RoboCasaCustomKitchen" \
  -n 4 -s 0 1 2 3 \
  --render-mode="human" -r "fetch"
```


**To add the object in the robocasa environment you should:**
1. create a robocasa environment.
2. use current conda environment and cd robocasa directory to use `pip install -e .`.
3. you can use the code `new_base_path = os.path.join(robocasa.models.assets_root, "objects")`.
4. you should `import os` and `import robocasa` at the top of the environment file.
