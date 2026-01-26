from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Optional
import warnings
from optuna.samplers import TPESampler
from ray import init, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.sample import Domain
import wandb
import yaml
from omg.datamodule import OMGDataModule
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer


def yield_flattened_items(d: dict):
    for key, value in d.items():
        if isinstance(value, dict):
            # Append parent key to child key with dot notation.
            for child_key, child_value in yield_flattened_items(value):
                yield f"{key}.{child_key}", child_value
        else:
            yield key, value


def overwrite_flattened_items(original_dict: dict, flattened_updates: dict) -> dict:
    new_dict = deepcopy(original_dict)
    for key, value in flattened_updates.items():
        assert not isinstance(value, dict)
        keys = key.split(".")
        current_dict = new_dict
        for k in keys[:-1]:
            assert k in current_dict
            current_dict = current_dict[k]
        assert keys[-1] in current_dict
        current_dict[keys[-1]] = value
    return new_dict


def validate_omg_tune(config: dict, base_omg_config: dict, ckpt_path: Path, project_name: str,
                      cpus_per_trial: int) -> None:
    context = tune.get_context()
    trial_dir = context.get_trial_dir()
    omg_config = overwrite_flattened_items(base_omg_config, config)
    omg_config_path = trial_dir + "/omg_config.yaml"
    with open(omg_config_path, "w") as f:
        yaml.safe_dump(omg_config, f)

    # Ignore specific warning from LightningCLI.
    warnings.filterwarnings(
        "ignore",
        "LightningCLI's args parameter is intended to run from within Python like if it were from the command line.*")
    # Pass only omg arguments to OMGCLI.
    # Run the 'load' subcommand that does nothing except loading the model, the datamodule, and optionally a checkpoint.
    # Using run=False would not work because it also disables loading from checkpoints.
    OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer, run=True,
           args=["validate", "--config", str(omg_config_path), "--ckpt_path", str(ckpt_path),
                 "--trainer.logger", "WandbLogger", "--trainer.logger.name", context.get_trial_name(),
                 "--trainer.logger.project", project_name,
                 "--seed_everything", "0", "--model.validation_mode", "metre",
                 "--model.store_validation_structures_path", trial_dir + "/val.xyz",
                 "--model.number_cpus", cpus_per_trial])
    wandb.teardown()


def tune_omg(num_samples: int, omg_config: Path, omg_ckpt_path: Path, storage_path: Path, temp_dir: Optional[Path],
             project_name: str, cpus_per_trial: int, gpus_per_trial: int, restore: bool, metric: str, mode: str) -> None:
    with open(omg_config, "r") as f:
        omg_config = yaml.unsafe_load(f)

    search_space = {}
    for key, value in yield_flattened_items(omg_config):
        if isinstance(value, Domain):
            if key in search_space:
                raise RuntimeError(f"Duplicate key {key} found in search space.")
            search_space[key] = value

    init(address="local", log_to_driver=True, _temp_dir=str(temp_dir) if temp_dir is not None else None)
    sampler = TPESampler(seed=0)
    # noinspection PyTypeChecker
    algo = OptunaSearch(sampler=sampler, metric=metric, mode=mode)
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    tune_func = tune.with_parameters(
        validate_omg_tune,
        base_omg_config=omg_config,
        ckpt_path=omg_ckpt_path,
        project_name=project_name,
        cpus_per_trial=cpus_per_trial,
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
    parser.add_argument("--omg_config", type=Path, required=True)
    parser.add_argument("--omg_ckpt_path", type=Path, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--storage_path", type=Path, default=Path("./"))
    parser.add_argument("--cpus_per_trial", type=int, default=4)
    parser.add_argument("--gpus_per_trial", type=int, default=1)
    parser.add_argument("--temp_dir", type=Path, default=None)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    absolute_omg_config_path = args.omg_config.absolute()
    if not absolute_omg_config_path.exists():
        raise RuntimeError(f"OMG config path {absolute_omg_config_path} does not exist.")

    absolute_omg_ckpt_path = args.omg_ckpt_path.absolute()
    if not absolute_omg_ckpt_path.exists():
        raise RuntimeError(f"OMG checkpoint path {absolute_omg_ckpt_path} does not exist.")

    absolute_storage_path = args.storage_path.absolute()
    if not absolute_storage_path.exists():
        raise RuntimeError(f"Storage path {absolute_storage_path} does not exist.")

    absolute_temp_dir = args.temp_dir.absolute() if args.temp_dir is not None else None
    if absolute_temp_dir is not None and not absolute_temp_dir.exists():
        raise RuntimeError(f"Temporary directory path {absolute_temp_dir} does not exist.")

    tune_omg(num_samples=-1, omg_config=absolute_omg_config_path, omg_ckpt_path=absolute_omg_ckpt_path,
             storage_path=absolute_storage_path, temp_dir=args.temp_dir, project_name=args.project_name,
             cpus_per_trial=args.cpus_per_trial, gpus_per_trial=args.gpus_per_trial, restore=args.restore,
             metric="corr_rmsd", mode="min")


if __name__ == '__main__':
    main()
