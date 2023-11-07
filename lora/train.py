import dataclasses
import datetime
import gc
import json
import math
import os
from dataclasses import dataclass

import click
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from baichuan2.model import BaichuanConfig, BaichuanForCausalLM
from baichuan2.tokenizer import BaichuanTokenizer
from lora.dataset import BelleChatDataset
from lora.model import LoraConfig, LoraModel


@dataclass
class TrainingConfig:
    # train
    num_epochs: int = 2
    batch_size: int = 2
    micro_batch_size: int = 4
    min_save_step: int = 20
    max_save_loss: float = 10.0

    # optimizer
    lr: float = 2e-5
    weight_decay: float = 0.1

    # scheduler
    gamma: float = 0.1


class Trainer:
    def __init__(
        self,
        project,
        dataset,
        base_model,
        tokenizer,
        lora_config,
        training_config,
        output_dir,
    ):
        self.project = project
        self.name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.dataset = dataset

        self.lora_config = lora_config
        self.lora_model = LoraModel(base_model=base_model, config=self.lora_config)
        self.tokenizer = tokenizer
        self.training_config = training_config

        self.config = {
            **dataclasses.asdict(lora_config),
            **dataclasses.asdict(training_config),
        }

        self.optimizer = AdamW(
            self.lora_model.parameters(),
            lr=self.training_config.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.training_config.weight_decay,
        )

        self.output_dir = output_dir
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.training_config.gamma
        )

    def train(self, use_wandb=True):
        if use_wandb:
            wandb.init(project=self.project, name=self.name, config=self.config)
            wandb.watch(self.lora_model, log="all")

        self.lora_model.train()

        step = 0
        loss = 100
        token_cnt = 0
        last_save_loss = None
        last_save_step = 0

        for epoch in range(self.training_config.num_epochs):
            loader = DataLoader(
                self.dataset, batch_size=self.training_config.batch_size
            )
            it = iter(loader)

            batch_size = (
                self.training_config.batch_size * self.training_config.micro_batch_size
            )
            one_epoch_steps = int(len(self.dataset) / batch_size)

            for _ in tqdm(range(one_epoch_steps)):
                lr = self.optimizer.param_groups[0]["lr"]

                self.optimizer.zero_grad(set_to_none=True)
                loss = 0

                for _ in range(self.training_config.micro_batch_size):
                    input_ids, labels, mask = next(it)
                    token_cnt += int(torch.count_nonzero(input_ids))
                    logits = self.lora_model.forward(input_ids, mask)
                    l = F.cross_entropy(
                        logits.view(-1, self.tokenizer.vocab_size()),
                        labels.view(-1),
                    )
                    if not math.isnan(l):
                        micro_loss = l / self.training_config.micro_batch_size
                        loss += micro_loss
                        micro_loss.backward()

                if use_wandb:
                    wandb.log(
                        {"loss": loss, "token_cnt": token_cnt, "learning_rate": lr},
                        step=step,
                    )

                if last_save_loss is None or loss < last_save_loss:
                    if step - last_save_step >= self.training_config.min_save_step:
                        if loss <= self.training_config.max_save_loss:
                            self.save(step, loss)
                            last_save_step = step
                            last_save_loss = loss

                self.optimizer.step()
                step += 1

            self.lr_scheduler.step()

        if use_wandb:
            wandb.finish()

    def save(self, step, loss):
        directory = os.path.join(
            self.output_dir, self.name, f"step={step}_loss={loss:.3f}"
        )
        os.makedirs(directory, exist_ok=True)

        state = {
            key: tensor
            for key, tensor in self.lora_model.state_dict().items()
            if "lora_" in key
        }
        ck_path = directory + "/weights.pt"
        torch.save(state, ck_path)

        with open(directory + "/training_config.json", "w") as o:
            c = json.dumps(dataclasses.asdict(self.training_config))
            json.dump(c, o)

        with open(directory + "/lora_config.json", "w") as o:
            c = json.dumps(dataclasses.asdict(self.lora_config))
            json.dump(c, o)


def train(
    project,
    vocab_file,
    checkpoint_dir,
    data_json_path,
    output_dir,
):
    # tokenizer
    tokenizer = BaichuanTokenizer(vocab_file=vocab_file)

    # load base model
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    model_config = BaichuanConfig()
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

    # dataset
    dataset = BelleChatDataset(
        tokenizer=tokenizer,
        data_json_path=data_json_path,
        model_max_length=1024,
    )

    # trainer
    trainer = Trainer(
        project,
        dataset,
        base_model,
        tokenizer,
        LoraConfig(),
        TrainingConfig(),
        output_dir,
    )
    trainer.train()


@click.command()
@click.option("project", "--project", required=True)
@click.option("vocab_file", "--vocab_file", required=True)
@click.option("checkpoint_dir", "--checkpoint_dir", required=True)
@click.option("data_json_path", "--data_json_path", required=True)
@click.option("output_dir", "--output_dir", required=True)
def main(
    project,
    vocab_file,
    checkpoint_dir,
    data_json_path,
    output_dir,
):
    train(project, vocab_file, checkpoint_dir, data_json_path, output_dir)


if __name__ == "__main__":
    main()
