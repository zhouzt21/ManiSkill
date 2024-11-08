import yaml
from collections import deque
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PImage

from embodied_agent.third_party.vla.rdt.constants import *
from embodied_agent.third_party.vla.rdt.scripts.agilex_model import (
    RoboticDiffusionTransformerModel, create_model as create_RDT_model
)


# TODO: currently, it do not support run in parallel

ALOHA_CAMERA_NAMES = [
    "mobile_aloha_base_high_camera", 
    "mobile_aloha_right_camera",
    "mobile_aloha_left_camera",
]

def dict_apply(d, func):
    """Applies `func` to all tensor leaves in a nested dictionary `d`."""
    if isinstance(d, dict):
        # Recursively apply to each element in the dictionary
        return {k: dict_apply(v, func) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        # Apply the function to the tensor leaf
        return func(d)
    else:
        # Return other types as-is
        return d

class RDTActor:
    def __init__(self, 
                 n_frames: int=2,
                 ctrl_freq: int=25, 
                 camera_names: List[str]=None, 
                 model_cfg_path: str=None,
                 device: str='cuda'):
        self.n_frames = n_frames
        self.ctrl_freq = ctrl_freq
        self.device = device
        self.model_cfg_path = RDT_DEFAULT_CONFIG if model_cfg_path is None else model_cfg_path

        self.obs_window = None
        self.lang_embeddings = None

        if camera_names is None:
            self.camera_names = deepcopy(ALOHA_CAMERA_NAMES)

        self.rdt_policy = self.make_policy()

    def make_policy(self):
        with open(RDT_DEFAULT_CONFIG, "r") as fp:
            config = yaml.safe_load(fp)

        model = create_RDT_model(
            args=config,
            dtype=torch.bfloat16,
            pretrained=RDT1B_PATH,
            pretrained_vision_encoder_name_or_path=SIGLIP_PATH,
            control_frequency=self.ctrl_freq
        )

        return model
    
    def update_obs_window(self, obs):
        if self.obs_window is None:
            self.obs_window = deque(maxlen=self.n_frames)

            self.obs_window.append(
                {
                    'qpos': None,
                    'images':
                        {
                            cam_name: None 
                                for cam_name in self.camera_names
                        }
                }
            )

        self.obs_window.append(
            {
                'qpos': obs['agent']['qpos'],
                'images':
                    {
                        cam_name: obs['sensor_data'][cam_name]['rgb'].permute(2, 0, 1)
                            for cam_name in self.camera_names
                    }
            }
        )

    @torch.no_grad()
    def encode_lang(self, lang_text):
        pass
    
    # RDT inference
    @torch.no_grad()
    def infer(self, lang_text=None, lang_embed=None):
        # fetch images in sequence [front, right, left]
        image_arrs = []
        for t in range(-self.n_frames, 0):
            for cam_name in self.camera_names:
                image_arrs.append(
                    self.obs_window[t]['images'][cam_name]
                )
        
        to_pil = transforms.ToPILImage()
        images = [to_pil(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # get last qpos in shape [14, ] and unsqueeze to [1, 14]
        proprio = self.obs_window[-1]['qpos']
        proprio = proprio.unsqueeze(0)

        lang_embeddings_path = '/home/share/rdt/lang_emb/handover_pan.pt'
        text_embedding = torch.load(lang_embeddings_path)['embeddings']  

        actions = self.rdt_policy.step(
            proprio=proprio,
            images=images,
            text_embeds=text_embedding
        )

        return actions

    def predict_action(self, obs):
        self.update_obs_window(
            dict_apply(obs, lambda x: torch.squeeze(x, dim=0)))
        actions = self.infer()

        return actions


        