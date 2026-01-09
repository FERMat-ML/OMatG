from typing import Dict, Set
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class OMGCLI(LightningCLI):
    """
    Command line interface for the omg package.

    Extends the LightningCLI class to add subcommands and argument linking specific to omg.

    Any initialization args and kwargs are passed down to the LightningCLI constructor.

    :param args:
        Positional arguments to pass to the LightningCLI constructor.
    :param kwargs:
        Keyword arguments to pass to the LightningCLI constructor.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the OMGCLI class."""
        super().__init__(*args, **kwargs)

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
        d["visualize"] = {"model", "datamodule"}
        d["csp_metrics"] = {"model", "datamodule"}
        d["dng_metrics"] = {"model", "datamodule"}
        d["fit_lattice"] = {"model", "datamodule"}
        d["create_compositions"] = {"model", "datamodule"}
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
                              "data.train_dataset.init_args.floating_point_precision")
        parser.link_arguments("trainer.precision",
                                "data.val_dataset.init_args.floating_point_precision")
        parser.link_arguments("trainer.precision",
                                "data.pred_dataset.init_args.floating_point_precision")
