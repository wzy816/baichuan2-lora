import time
from dataclasses import replace
from itertools import product
from functools import reduce

import click

from lora.train import train
from lora.util import load_lora_config


@click.command()
@click.option("project", "--project", required=True)
@click.option("vocab_file", "--vocab_file", required=True)
@click.option("checkpoint_dir", "--checkpoint_dir", required=True)
@click.option("data_json_path", "--data_json_path", required=True)
@click.option("config_yaml_path", "--config_yaml_path", required=False)
@click.option("output_dir", "--output_dir", required=True)
def main(
    project,
    vocab_file,
    checkpoint_dir,
    data_json_path,
    config_yaml_path,
    output_dir,
):
    params = {
        "r": [16,32,64,128],
        "dropout": [0.01],
        "alpha": [16,32,64,128],
        "lr": [2e-3,2e-4,2e-5],
        "num_epochs": [1],
        "batch_size": [2],
        "micro_batch_size": [100],
        "num_samples": [40000]
    }
    combinations = product(*params.values())
    total_combinations = reduce(lambda x, y: x * y, [len(v) for v in params.values()])
    
    for idx, values in enumerate(combinations):
        config = {}
        for k, v in zip(params.keys(), values):
            config[k] = v

        if config['r'] / config['alpha'] > 2 or config['r'] / config['alpha'] < 1/2:
            continue
        lora_config = replace(load_lora_config(config_yaml_path), **config)
        print(f'combo {idx} / {total_combinations}',lora_config)

        train(
            project,
            vocab_file,
            checkpoint_dir,
            data_json_path,
            lora_config,
            output_dir,
            use_wandb=True,
            use_tqdm=True,
            wandb_mode='offline',
            shuffle_dataset=False,
        )
        time.sleep(20)


if __name__ == "__main__":
    main()
