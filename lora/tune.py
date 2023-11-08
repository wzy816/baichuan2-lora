from dataclasses import replace

import click
import torch
from ray import train as ray_train
from ray import tune

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
    def objective(config):
        assert torch.cuda.is_available()
        lora_config = load_lora_config(config_yaml_path)
        lora_config = replace(lora_config, **config)

        last_save_loss = train(
            project,
            vocab_file,
            checkpoint_dir,
            data_json_path,
            lora_config,
            output_dir,
            use_wandb=True,
            use_tqdm=False,
        )
        ray_train.report({"last_save_loss": last_save_loss})

    config = {
        "r": tune.grid_search([1, 4, 16, 64]),
        "dropout": tune.grid_search([0.1, 0.05, 0]),
        "alpha": tune.grid_search([32, 64]),
        "lr": tune.grid_search([2e-5, 2e-4]),
        "num_epochs": 1,
        "batch_size": 2,
        "micro_batch_size": 4,
    }

    tuner = tune.Tuner(tune.with_resources(objective, {"gpu": 1}), param_space=config)
    results = tuner.fit()
    print(results.get_best_result().config)


if __name__ == "__main__":
    main()
