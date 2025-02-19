import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import tyro
import os 
import time 

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 78
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    lr: float = 3e-4
    """the learning rate for the actor"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # setup
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"./runs/{run_name}/videos" if args.capture_video else None,
    )

    obs, _ = envs.reset(seed=args.seed)

    # load checkpoint
    checkpoint_path = './runs/PushCube-v1__bc__1__1738744733/checkpoints/best_eval_success_at_end.pt'
    num_eval_episodes = 10

    actor = Actor(
        envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    )
    actor.load_state_dict(torch.load(checkpoint_path)['actor'])
    actor = actor.to(device=device)

    optimizer = optim.Adam(actor.parameters(), lr=args.lr)  # is that neccessary?

    best_eval_metrics = defaultdict(float)

    # evaluation
    def sample_fn(obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        action = actor(obs)
        if args.sim_backend == "cpu":
            action = action.cpu().numpy()
        return action

    actor.eval()
    with torch.no_grad():
        eval_metrics = evaluate(num_eval_episodes, sample_fn, envs)

    envs.close()