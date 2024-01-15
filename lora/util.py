import gc
import json
import os
from typing import Optional

import torch
import yaml
from tqdm import tqdm

from baichuan2.model import BaichuanConfig, BaichuanForCausalLM
from lora.config import LoraConfig


def load_base_model(
    checkpoint_dir,
    dtype,
    model_config: BaichuanConfig = BaichuanConfig(),
) -> BaichuanForCausalLM:
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    base_model = BaichuanForCausalLM(model_config)

    shard_files = [
        os.path.join(checkpoint_dir, f"pytorch_model-0000{i+1}-of-00003.bin")
        for i in range(3)
    ]

    for f in tqdm(shard_files, desc="loading model shards"):
        assert os.path.exists(f)

        state_dict = torch.load(f, map_location="cuda:0")
        base_model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
    return base_model


def load_lora_config(config_path: Optional[str] = None) -> LoraConfig:
    if config_path is not None:
        if config_path.endswith(".yaml"):
            with open(config_path, "r", encoding="utf8") as f:
                d = yaml.safe_load(f)
                return LoraConfig(**d)

        elif config_path.endswith(".json"):
            j = json.load(open(config_path, "r", encoding="utf8"))
            d = json.loads(j)
            return LoraConfig(**d)

        else:
            raise Exception("unsupported config format")
    else:
        return LoraConfig()
