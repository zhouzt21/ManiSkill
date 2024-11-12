# Troubleshooting

## Mobile Aloha

1. Load robot

For using mobile aloha, you need to unzip `mani_skill/assets/robots/mobile_aloha/meshes/box2_Link.dae.zip` to obtain the required `.dae` file. The original file is ignored in git because of its large size.

2. `python -m demo_mobile_aloha`

Since some of the provided model files for mobile aloha are highly detailed, certain models cannot be automatically converted to convex shapes by Sapien. However, convexity does not impact the simulation. Therefore, a small modification is needed in the `init` method of the `mplib` planner class. After installing `mplib` via pip, locate `mplib/planner.py` in your conda environment and change line 71 from `convex=True` to `convex=False`.

Note: The typical file path is `{path to your conda}/envs/maniskill/lib/python3.10/site-packages/mplib/planner.py`.
