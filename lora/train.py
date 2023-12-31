import dataclasses
import datetime
import json
import math
import os

import click
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from baichuan2.model import BaichuanConfig
from baichuan2.tokenizer import BaichuanTokenizer
from lora.config import LoraConfig
from lora.dataset import ChatDataset
from lora.model import LoraModel
from lora.util import load_base_model, load_lora_config


class Trainer:
    def __init__(
        self,
        project,
        dataset,
        base_model,
        tokenizer,
        lora_config,
        output_dir,
    ):
        self.project = project
        self.name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.dataset = dataset

        self.lora_config = lora_config
        self.lora_model = LoraModel(base_model=base_model, config=self.lora_config)
        self.tokenizer = tokenizer

        self.optimizer = AdamW(
            self.lora_model.parameters(),
            lr=self.lora_config.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.lora_config.weight_decay,
        )

        self.output_dir = output_dir
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.lora_config.gamma
        )
        self.last_save_loss = None
        self.last_save_step = 0
        self.best_loss = math.inf

    def train(self, use_wandb: bool, use_tqdm: bool, wandb_mode: str):
        if use_wandb:
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.lora_config,
                mode=wandb_mode,
            )
            wandb.watch(self.lora_model, log="all")

        self.lora_model.train()

        step = 0
        token_cnt = 0

        for epoch in range(self.lora_config.num_epochs):
            loader = DataLoader(self.dataset, batch_size=self.lora_config.batch_size)
            it = iter(loader)

            if self.lora_config.num_samples == -1:
                num_samples = len(self.dataset)
            else:
                num_samples = self.lora_config.num_samples

            epoch_steps = int(
                num_samples
                / self.lora_config.batch_size
                / self.lora_config.micro_batch_size
            )

            epoch_steps = range(epoch_steps)
            if use_tqdm:
                epoch_steps = tqdm(epoch_steps, desc=f"epoch {epoch}")

            for _ in epoch_steps:
                lr = self.optimizer.param_groups[0]["lr"]

                self.optimizer.zero_grad(set_to_none=True)
                loss = 0

                for _ in range(self.lora_config.micro_batch_size):
                    input_ids, labels, mask = next(it)
                    token_cnt += int(torch.count_nonzero(input_ids))
                    logits = self.lora_model.forward(input_ids, mask)
                    l = F.cross_entropy(
                        logits.view(-1, self.tokenizer.vocab_size()),
                        labels.view(-1),
                    )
                    if not math.isnan(l):
                        micro_loss = l / self.lora_config.micro_batch_size
                        loss += micro_loss
                        micro_loss.backward()

                if use_wandb:
                    wandb.log(
                        {"loss": loss, "token_cnt": token_cnt, "learning_rate": lr},
                        step=step,
                    )

                if self.last_save_loss is None or loss < self.last_save_loss:
                    if step - self.last_save_step >= self.lora_config.min_save_step:
                        if loss <= self.lora_config.max_save_loss:
                            self.save(step, loss)
                            self.last_save_step = step
                            self.last_save_loss = loss

                if loss < self.best_loss:
                    self.best_loss = loss

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

        with open(directory + "/lora_config.json", "w", encoding="utf-8") as o:
            c = json.dumps(dataclasses.asdict(self.lora_config))
            json.dump(c, o)


def train(
    project: str,
    vocab_file: str,
    checkpoint_dir: str,
    data_json_path: str,
    lora_config: LoraConfig,
    output_dir: str,
    use_wandb: bool = True,
    use_tqdm: bool = True,
    wandb_mode: str = "online",
    shuffle_dataset: bool = True,
):
    tokenizer = BaichuanTokenizer(vocab_file=vocab_file)

    model_config = BaichuanConfig()
    base_model = load_base_model(checkpoint_dir, torch.bfloat16, model_config)

    dataset = ChatDataset(
        tokenizer=tokenizer,
        data_json_path=data_json_path,
        model_max_length=1024,  # model_config.model_max_length
        shuffle_dataset=shuffle_dataset,
    )

    trainer = Trainer(
        project,
        dataset,
        base_model,
        tokenizer,
        lora_config,
        output_dir,
    )
    trainer.train(
        use_wandb=use_wandb,
        use_tqdm=use_tqdm,
        wandb_mode=wandb_mode,
    )


@click.command()
@click.option("project", "--project", required=True)
@click.option("vocab_file", "--vocab_file", required=True)
@click.option("checkpoint_dir", "--checkpoint_dir", required=True)
@click.option("data_json_path", "--data_json_path", required=True)
@click.option("config_path", "--config_path", required=False)
@click.option("output_dir", "--output_dir", required=True)
def main(
    project,
    vocab_file,
    checkpoint_dir,
    data_json_path,
    config_path,
    output_dir,
):
    lora_config = load_lora_config(config_path)
    train(
        project,
        vocab_file,
        checkpoint_dir,
        data_json_path,
        lora_config,
        output_dir,
    )


if __name__ == "__main__":
    main()
