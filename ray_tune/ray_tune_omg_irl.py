from argparse import ArgumentParser
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional
import warnings
from optuna.samplers import TPESampler
from ray import init, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.sample import Domain
import wandb
import yaml
from omg.datamodule import OMGDataModule
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer
from omg.omg_irl.omg_irl_trainer import OMGIRLTrainer


class LimitTrainer(OMGIRLTrainer):  # TODO: MAKE SURE THESE FILES DON'T APPEAR IN PACKAGE
    # On some clusters, we have to specifically enforce int type if we want to set val_check_interval.
    def __init__(self, val_check_interval: int, *args, **kwargs):
        # Add limit train_batches to kwargs.
        kwargs["val_check_interval"] = val_check_interval
        super().__init__(*args, **kwargs)


def yield_flattened_items(d: dict, prefix: tuple = ()):
    for key, value in d.items():
        current_path = prefix + (key,)
        if isinstance(value, dict):
            yield from yield_flattened_items(value, current_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                item_path = current_path + (index,)
                if isinstance(item, dict):
                    yield from yield_flattened_items(item, item_path)
                else:
                    yield item_path, item
        else:
            yield current_path, value


def overwrite_flattened_items(original_dict: dict, flattened_updates: dict[tuple, Any]) -> dict:
    new_dict = deepcopy(original_dict)
    for key, value in flattened_updates.items():
        keys = key.split(".")
        current = new_dict
        for k in keys[:-1]:
            try:
                index = int(k)
                current = current[index]
            except ValueError:
                current = current[k]
        final_key = keys[-1]
        try:
            index = int(final_key)
            current[index] = value
        except ValueError:
            current[final_key] = value
    return new_dict


def train_omg_irl_tune(config: dict, base_rl_config: dict, base_omg_config_path: Path, ckpt_path: Path,
                       project_name: str, omg_irl_ckpt_path: Optional[Path]):
    # Import here to avoid issues with global base_modules when using Ray Tune.
    from omg.omg_irl.base_modules import base_modules
    from omg.omg_irl.omg_irl_lightning.abstracts import OMGIRLLightningAbstract
    from omg.omg_irl.omg_irl_cli import OMGIRLCLI

    context = tune.get_context()
    trial_dir = context.get_trial_dir()
    rl_config = overwrite_flattened_items(base_rl_config, config)
    rl_config_path = trial_dir + "/rl_config.yaml"
    with open(rl_config_path, "w") as f:
        yaml.safe_dump(rl_config, f)

    try:
        # Ignore specific warning from LightningCLI.
        warnings.filterwarnings(
            "ignore",
            "LightningCLI's args parameter is intended to run from within Python like if it were from the command line.*")
        # Pass only omg arguments to OMGCLI.
        # Run the 'load' subcommand that does nothing except loading the model, the datamodule, and optionally a checkpoint.
        # Using run=False would not work because it also disables loading from checkpoints.
        omg_cli = OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer,
                         save_config_callback=None, run=True,
                         args=["load", "--config", str(base_omg_config_path), "--ckpt_path", str(ckpt_path)])
        # Move OMG model to the correct device.
        # The sequence of function calls is taken from the _run method in the Lightning Trainer.
        trainer = omg_cli.trainer
        trainer.strategy.connect(omg_cli.model)
        trainer.strategy.setup_environment()
        trainer.strategy.setup(omg_cli.trainer)
        # Set global base models and datamodule.
        base_modules["model"] = omg_cli.model
        base_modules["datamodule"] = omg_cli.datamodule

        if omg_irl_ckpt_path is None:
            OMGIRLCLI(model_class=OMGIRLLightningAbstract, trainer_class=LimitTrainer,
                      save_config_callback=None, subclass_mode_model=True,
                      args=["fit", "--config", str(rl_config_path), "--trainer.logger", "WandbLogger",
                            "--trainer.logger.name", context.get_trial_name(), "--trainer.logger.project", project_name,
                            "--seed_everything", "0", "--model.init_args.validation_xyz_filename",
                            trial_dir + "/val.xyz"])
        else:
            OMGIRLCLI(model_class=OMGIRLLightningAbstract, trainer_class=LimitTrainer,
                      save_config_callback=None, subclass_mode_model=True,
                      args=["fit", "--config", str(rl_config_path), "--ckpt_path", str(omg_irl_ckpt_path),
                            "--trainer.logger", "WandbLogger", "--trainer.logger.name", context.get_trial_name(),
                            "--trainer.logger.project", project_name, "--seed_everything", "0",
                            "--model.init_args.validation_xyz_filename", trial_dir + "/val.xyz"])

    finally:
        # Necessary to flush stdout and stderr files.
        sys.stdout.flush()
        sys.stderr.flush()
        # For some reason, these files are not copied over automatically.
        # This part will only run on errors, not when the ASHAScheduler kills a trial.
        try:
            shutil.copy(f"{os.getcwd()}/stdout", f"{trial_dir}/")
        except FileNotFoundError:
            pass
        try:
            shutil.copy(f"{os.getcwd()}/stderr", f"{trial_dir}/")
        except FileNotFoundError:
            pass
        wandb.teardown()


def tune_omg_irl(num_samples: int, rl_config: Path, omg_config: Path, omg_ckpt_path: Path,
                 storage_path: Path, temp_dir: Optional[Path], project_name: str, cpus_per_trial: int,
                 gpus_per_trial: int, restore: bool, metric: str, mode: str, max_t: int, grace_period: int,
                 omg_irl_ckpt_path: Optional[Path]) -> None:
    with open(rl_config, "r") as f:
        rl_config = yaml.unsafe_load(f)

    search_space = {}
    for key, value in yield_flattened_items(rl_config):
        if isinstance(value, Domain):
            # Convert tuple key to dot-separated string
            str_key = ".".join(str(k) for k in key)
            if str_key in search_space:
                raise RuntimeError(f"Duplicate key {str_key} found in search space.")
            search_space[str_key] = value

    init(address="local", log_to_driver=True, _temp_dir=str(temp_dir) if temp_dir is not None else None)
    # Here max_t is in unites of the computation of the metric.
    scheduler = ASHAScheduler(max_t=max_t, grace_period=grace_period)
    sampler = TPESampler(seed=0)
    # noinspection PyTypeChecker
    algo = OptunaSearch(sampler=sampler, metric=metric, mode=mode)
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    tune_func = tune.with_parameters(
        train_omg_irl_tune,
        base_rl_config=rl_config,
        base_omg_config_path=omg_config,
        ckpt_path=omg_ckpt_path,
        project_name=project_name,
        omg_irl_ckpt_path=omg_irl_ckpt_path,
    )

    if restore:
        restore_path = storage_path / project_name
        print(f"Restoring from {restore_path}")
        tuner = tune.Tuner.restore(
            str(restore_path),
            tune.with_resources(tune_func, resources=resources_per_trial),
            param_space=search_space,
            resume_unfinished=False,
            restart_errored=False
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(tune_func, resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                num_samples=num_samples,
                scheduler=scheduler,
                search_alg=algo,
            ),
            param_space=search_space,
            run_config=tune.RunConfig(
                name=project_name,
                storage_path=str(storage_path)
            ),
        )
    tuner.fit()


def main():
    parser = ArgumentParser()
    parser.add_argument("--rl_config", type=Path, required=True)
    parser.add_argument("--omg_config", type=Path, required=True)
    parser.add_argument("--omg_ckpt_path", type=Path, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--storage_path", type=Path, default=Path("./"))
    parser.add_argument("--cpus_per_trial", type=int, default=4)
    parser.add_argument("--gpus_per_trial", type=int, default=1)
    parser.add_argument("--temp_dir", type=Path, default=None)
    parser.add_argument("--omg_irl_ckpt_path", type=Path, default=None)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--sde", action="store_true")
    args = parser.parse_args()

    absolute_rl_config_path = args.rl_config.absolute()
    if not absolute_rl_config_path.exists():
        raise RuntimeError(f"RL config path {absolute_rl_config_path} does not exist.")

    absolute_omg_config_path = args.omg_config.absolute()
    if not absolute_omg_config_path.exists():
        raise RuntimeError(f"OMG config path {absolute_omg_config_path} does not exist.")

    absolute_omg_ckpt_path = args.omg_ckpt_path.absolute()
    if not absolute_omg_ckpt_path.exists():
        raise RuntimeError(f"OMG checkpoint path {absolute_omg_ckpt_path} does not exist.")

    absolute_storage_path = args.storage_path.absolute()
    if not absolute_storage_path.exists():
        raise RuntimeError(f"Storage path {absolute_storage_path} does not exist.")

    if args.omg_irl_ckpt_path is not None:
        absolute_omg_irl_ckpt_path = args.omg_irl_ckpt_path.absolute()
        if not absolute_omg_irl_ckpt_path.exists():
            raise RuntimeError(f"OMG-IRL checkpoint path {absolute_omg_irl_ckpt_path} does not exist.")
    else:
        absolute_omg_irl_ckpt_path = None

    absolute_temp_dir = args.temp_dir.absolute() if args.temp_dir is not None else None
    if absolute_temp_dir is not None and not absolute_temp_dir.exists():
        raise RuntimeError(f"Temporary directory path {absolute_temp_dir} does not exist.")

    tune_omg_irl(num_samples=-1, rl_config=absolute_rl_config_path, omg_config=absolute_omg_config_path,
                 omg_ckpt_path=absolute_omg_ckpt_path, storage_path=absolute_storage_path, temp_dir=args.temp_dir,
                 project_name=args.project_name, cpus_per_trial=args.cpus_per_trial, gpus_per_trial=args.gpus_per_trial,
                 restore=args.restore, metric="val_reward_mean", mode="max", max_t=10, grace_period=5,
                 omg_irl_ckpt_path=absolute_omg_irl_ckpt_path)


if __name__ == '__main__':
    main()
