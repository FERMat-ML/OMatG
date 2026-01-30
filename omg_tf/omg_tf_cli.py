from typing import Dict, Set
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from omg_tf.base_modules import base_modules


class OMGTFCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the OMGTFCLI class."""
        super().__init__(*args, **kwargs)

    def after_instantiate_classes(self) -> None:
        """Hook to modify instantiated classes after instantiation."""
        super().after_instantiate_classes()
        if base_modules["datamodule"] is None:
            raise ValueError("Base datamodule must be set globally before instantiating OMGTFCLI.")
        self.datamodule = base_modules["datamodule"]

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """
        Defines additional available subcommands (see corresponding methods in the OMGTrainer class) and the
        arguments to skip.

        :return:
            A dictionary where keys are subcommand names and values are sets of argument names that should be skipped.
        :rtype: Dict[str, Set[str]]
        """
        d = LightningCLI.subcommands()
        d["plot_schedule"] = {"model", "datamodule"}
        return d

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """
        Link certain arguments in the YAML/CLI configuration so that only one of them has to be set.

        See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html.

        :param parser:
            The argument parser.
        :type parser: LightningArgumentParser
        """
        # TODO: If trainer.enable_progress_bar is not set, progress bar in trainer is on but model's is off.
        parser.link_arguments("trainer.enable_progress_bar", "model.enable_progress_bar")
