import yaml
from collections import deque
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PImage

from embodied_agent.third_party.vla.rdt.constants import *
from embodied_agent.third_party.vla.rdt.models.multimodal_encoder.t5_encoder import T5Embedder
from embodied_agent.third_party.vla.rdt.scripts.agilex_model import (
    RoboticDiffusionTransformerModel, create_model as create_RDT_model
)


# TODO: currently, it do not support batched execution.

ALOHA_CAMERA_NAMES = [
    "cam_high", 
    "cam_right_wrist",
    "cam_left_wrist",
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
                 ctrl_freq: int=25, 
                 camera_names: List[str]=None, 
                 model_cfg_path: str=None,
                 device: str='cuda'):
        self.ctrl_freq = ctrl_freq
        self.device = device
        self.model_cfg_path = RDT_DEFAULT_CONFIG if model_cfg_path is None else model_cfg_path

        self.obs_window = None
        self.lang_embeddings = None

        if camera_names is None:
            self.camera_names = deepcopy(ALOHA_CAMERA_NAMES)

        with open(self.model_cfg_path, "r") as fp:
            self.config = yaml.safe_load(fp)

        self.n_frames = self.config["common"]["img_history_size"]
        self.action_chunk_size = self.config["common"]["action_chunk_size"]

        self.rdt_policy = self.make_policy()
        self.lang_tokenizer, self.lang_encoder = self.make_lang_models()

        self.last_instruction = None
        self.action_buffer = None

    def make_policy(self):
        model = create_RDT_model(
            args=self.config,
            dtype=torch.bfloat16,
            pretrained=RDT1B_PATH,
            pretrained_vision_encoder_name_or_path=SIGLIP_PATH,
            control_frequency=self.ctrl_freq
        )

        return model
    
    def make_lang_models(self):
        # Note: if your GPU VRAM is less than 24GB, 
        # it is recommanded to enable offloading by specifying an offload directory.
        text_embedder = T5Embedder(
            from_pretrained=T5_PATH, 
            model_max_length=self.config["dataset"]["tokenizer_max_length"], 
            device=self.device,
            use_offload_folder=None # Specify your offload directory here, ensuring the directory exists.
        )

        return text_embedder.tokenizer, text_embedder.model
    
    @torch.no_grad()
    def encode_instruction(self, instr: str):
        token = self.lang_tokenizer(
            instr, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(self.device)

        tokens = tokens.view(1, -1)
        
        pred = self.lang_encoder(tokens).last_hidden_state

        return pred
    
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
                'qpos': torch.concat([obs['agent']['qpos_l'], obs['agent']['qpos_r']]),
                'images':
                    {
                        cam_name: obs['sensor_data'][cam_name]['rgb'].permute(2, 0, 1)
                            for cam_name in self.camera_names
                    }
            }
        )

    # RDT inference
    @torch.no_grad()
    def infer(self, lang_instruction: str):
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

        actions = self.rdt_policy.step(
            proprio=proprio,
            images=images,
            text_embeds=self.text_embedding
        )

        return actions

    def predict_action(self, obs: Dict, instr: str):
        if instr is not self.last_instruction:
            self.last_instruction = instr
            self.text_embedding = self.encode_instruction(instr)

        self.update_obs_window(
            dict_apply(obs, lambda x: torch.squeeze(x, dim=0)))
        actions = self.infer()

        return actions


        