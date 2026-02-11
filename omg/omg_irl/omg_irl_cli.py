from typing import Dict, Set
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from omg.omg_irl.base_modules import base_modules


class OMGIRLCLI(LightningCLI):
    """
    Command line interface for the omg_irl package.

    Extends the LightningCLI class to add subcommands and argument linking specific to omg_irl.

    Any initialization args and kwargs are passed down to the LightningCLI constructor.

    :param args:
        Positional arguments to pass to the LightningCLI constructor.
    :param kwargs:
        Keyword arguments to pass to the LightningCLI constructor.
    """
    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the OMGIRLCLI class."""
        super().__init__(*args, **kwargs)

    def after_instantiate_classes(self) -> None:
        """
        Hook to modify instantiated classes after instantiation.

        This hook is called automatically by LightningCLI. This allows us to set the datamodule attribute of the
        OMGIRLCLI class to the base datamodule that is set globally in the base_modules dictionary.
        """
        super().after_instantiate_classes()
        if base_modules["datamodule"] is None:
            raise ValueError("Base datamodule must be set globally before instantiating OMGIRLCLI.")
        self.datamodule = base_modules["datamodule"]

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """
        Defines additional available subcommands (see corresponding methods in the OMGIRLTrainer class) and the
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
        parser.link_arguments("trainer.enable_progress_bar", "model.init_args.enable_progress_bar")
