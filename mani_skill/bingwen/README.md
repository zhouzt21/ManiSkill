### Config
**create your own environment**:
- you can copy mani_skill/envs/tasks/mobile_manipulation/robocasa/kitchen.py to a new file,eg,mani_skill/envs/tasks/mobile_manipulation/robocasa/custom_kitchen.py. 
- you should rename the env_id in the @register_env() function.
- you should modify the __init__.py in the  directory. such as add `from .custom_kitchen import RoboCasaCustomKitchenEnv`

Then, you can use the command below in the bash

``` bash
  python -m mani_skill.examples.demo_random_action \
  -e "RoboCasaCustomKitchen" \
  -n 4 -s 0 1 2 3 \
  --render-mode="human" -r "fetch"
```
