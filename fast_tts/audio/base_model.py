# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:28
# Author    :Hui Huang
import json

import torch
import torch.nn as nn
from .utils import load_config
import os
from safetensors.torch import load_file


class SparkBaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, model_path: str):
        config = load_config(os.path.join(model_path, "config.yaml"))['audio_tokenizer']
        model = cls(config)
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.remove_weight_norm()
        return model

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


class SnacBaseModel(nn.Module):
    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, model_path: str):

        model = cls.from_config(os.path.join(model_path, "config.json"))
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
