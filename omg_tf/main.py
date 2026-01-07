import argparse
import sys
import warnings
from omg import __version__
from omg.datamodule import OMGDataModule
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer
from omg_tf.base_modules import base_modules
from omg_tf import omg_tf_lightning_ppo
from omg_tf import omg_tf_cli


def main():
    """Main function to run the Open Materials Generation with Targeted Functions (OMatG-TF) command line interface
    (used by omg_tf command)."""
    args = sys.argv[1:]
    try:
        omg_index = args.index("omg")
    except ValueError:
        # Suppress printing traceback.
        sys.exit("error: omg_tf requires configuration of omg model after 'omg' command-line argument (without any "
                 "subcommand)")

    if any(subcommand in args[omg_index + 1:] for subcommand in OMGCLI.subcommands().keys()):
        sys.exit("error: omg_tf requires configuration of omg model after 'omg' command-line argument (without any "
                 "subcommand)")

    # Ignore specific warning from LightningCLI.
    warnings.filterwarnings(
        "ignore",
        "LightningCLI's args parameter is intended to run from within Python like if it were from the command line.*")

    # Pass only omg arguments to OMGCLI.
    # Run the 'load' subcommand that does nothing except loading the model, the datamodule, and optionally a checkpoint.
    # Using run=False would not work because it also disables loading from checkpoints.
    omg_cli = OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer, run=True,
                     args=["load"] + args[omg_index + 1:])
    print("OMG model and datamodule loaded successfully.")

    # Move OMG model to the correct device.
    # The sequence of function calls is taken from the _run method in the Lightning Trainer.
    trainer = omg_cli.trainer
    trainer.strategy.connect(omg_cli.model)
    trainer.strategy.setup_environment()
    trainer.strategy.setup(omg_cli.trainer)
    # Set global base models and datamodule.
    base_modules["model"] = omg_cli.model
    base_modules["datamodule"] = omg_cli.datamodule

    # TODO: DOES HELP STILL WORK?

    # Pass only omg_tf arguments.
    omg_tf_cli.OMGTFCLI(model_class=omg_tf_lightning_ppo.OMGTFLightningPPO, args=args[:omg_index],
                        parser_kwargs={"formatter_class": argparse.RawDescriptionHelpFormatter, "description": f"""
Open Materials Generation (OMatG) Version {__version__}

A state-of-the-art generative model for crystal structure prediction and de novo generation of inorganic crystals.
"""})
