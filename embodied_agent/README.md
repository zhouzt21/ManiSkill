# Embodied Agent

## TODO

- [ ] add some single-stage tasks to verify RDT on skill execution
- [ ] add some multi-stage tasks to verify VLM on task decomposition

## Installation

### 1. Install ManiSkill-3, slightly modified from [the official documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)

```bash
conda create -n maniskill python=3.10 && conda activate maniskill
cd ManiSkill && pip install -e . # this step will also add `embodied_agent` to the python path
# install a version of torch that is compatible with your system
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
```

PS: Refer to [this instruction](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting) to fix vulkan.

### 2. Install RDT (Robotics Diffusion Transformer)

1. Install dependencies 

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

2. Download off-the-shelf multi-modal encoders

You can download the encoders from the following links:

- `t5-v1_1-xxl`: [link](https://huggingface.co/google/t5-v1_1-xxl/tree/main)🤗
- `siglip`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)🤗

And link the encoders to the repo directory:

```bash
# Under the root directory of this repo
mkdir -p google

# Link the downloaded encoders to this repo
ln -s /path/to/t5-v1_1-xxl google/t5-v1_1-xxl
ln -s /path/to/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
```

In this repo, we also store the pretrained weights for RDT in the same dir

- `RDT-1B`: [link](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)🤗
- `RDT-170M`: [link](https://huggingface.co/robotics-diffusion-transformer/rdt-170m)🤗

```bash
ln -s /path/to/rdt-1b google/rdt-1b
ln -s /path/to/rdt-170m google/rdt-170m
```

## Demo

After installation, you can run the following command to verify your installation which will save the episode into `video` folder in your current dir.

```bash
python -m embodied_agent.examples.demo_rdt --record_dir 'video' --render_mode sensors
```

