from dataclasses import dataclass


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_module: str = "W_pack"

    # train
    num_epochs: int = 2
    batch_size: int = 2
    micro_batch_size: int = 4
    num_samples: int = -1

    # save
    min_save_step: int = 20
    max_save_loss: float = 10.0

    # optimizer
    lr: float = 2e-5
    weight_decay: float = 0.1

    # scheduler
    gamma: float = 0.1