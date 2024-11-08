# Embodied Agent

## Installation

1. Install ManiSkill-3, slightly modified from [the official documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)

```bash
conda create -n maniskill python=3.10 && conda activate maniskill
cd ManiSkill && pip install -e . # this step will also add `embodied_agent` to the python path
# install a version of torch that is compatible with your system
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
```

PS: Refer to [this instruction](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting) to fix vulkan.

1. Install dependencies for RDT (Robotics Diffsuion Transformer)

```bash
# in the embodied_agent dir

# RDT has been added as a git submodule, which was modified for compatibility.
git submodule update --init --recursive 

# Install packaging
pip install packaging==24.0

# install a specific-version setuptools to avoid errors:
#   _DeprecatedInstaller: setuptools.installer and fetch_build_eggs are deprecated.
pip install -U setuptools==75.1.0

# Install flash-attn
pip install flash-attn --no-build-isolation

# Install other prequisites
pip install -r third_party/vla/rdt/requirements.txt
```
