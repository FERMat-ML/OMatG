from typing import Dict, Set
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class OMGCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        d = LightningCLI.subcommands()
        d["visualize"] = {"model", "datamodule"}
        d["match"] = {"model", "datamodule"}
        d["curriculum"] = {"model", "datamodule"}
        return d

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """
        Link certain arguments in the YAML/CLI configuration so that only one of them has to be set.

        See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html.

        :param parser:
            The argument parser.
        :type parser: LightningArgumentParser
        """
        parser.link_arguments("data.batch_size", "model.sampler.init_args.batch_size")
        parser.link_arguments("trainer.precision",
                              "data.train_dataset.init_args.dataset.init_args.trainer_precision")
        parser.link_arguments("trainer.precision",
                                "data.val_dataset.init_args.dataset.init_args.trainer_precision")
        parser.link_arguments("trainer.precision",
                                "data.predict_dataset.init_args.dataset.init_args.trainer_precision")
