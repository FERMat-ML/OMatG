import argparse
from typing import Any, Sequence
import huggingface_hub
from omg import __version__
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer
from omg.datamodule import OMGDataModule


def main():
    """Main function to run the Open Materials Generation (OMatG) command line interface (used by omg command)."""
    OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer,
           parser_kwargs={"formatter_class": argparse.RawDescriptionHelpFormatter, "description": f"""
Open Materials Generation (OMatG) Version {__version__}

A state-of-the-art generative model for crystal structure prediction and de novo generation of inorganic crystals.
"""})


class ListDatasetsAction(argparse.Action):
    """Custom argparse action to list available OMatG datasets from Hugging Face and exit."""
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: str | Sequence[Any] | None, option_string: str | None = None) -> None:
        """List available OMatG datasets from Hugging Face and exit."""
        print("""\
Available OMatG datasets on Hugging Face:
=========================================
""")
        for ds in huggingface_hub.list_datasets(author="OMatG"):
            print(ds.id)
        parser.exit(0)


def load():
    """Load an OMatG dataset from Hugging Face (used by omg_load command)."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
Open Materials Generation (OMatG) Version {__version__}

Load OMatG datasets from Hugging Face (see https://huggingface.co/OMatG/datasets).
The dataset will be downloaded to a local directory named after the dataset.
""")
    parser.add_argument("dataset_name", type=str,
                        help="name of the OMatG dataset to load from Hugging Face")
    parser.add_argument("-l", "--list", nargs=0, action=ListDatasetsAction,
                        help="list all available OMatG datasets on Hugging Face and exit")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if not dataset_name.startswith("OMatG/"):
        dataset_name = f"OMatG/{dataset_name}"
    print(f"Loading dataset '{dataset_name}' from Hugging Face.")

    dataset_name_only = dataset_name.split("OMatG/")[-1]

    try:
        huggingface_hub.snapshot_download(dataset_name, repo_type="dataset", allow_patterns="*.parquet",
                                          local_dir=dataset_name_only, cache_dir=dataset_name_only)
    except huggingface_hub.errors.RepositoryNotFoundError as e:
        # Modify message to suggest using the -l/--list option.
        e.args = (f"Dataset '{dataset_name}' not found on Hugging Face. "
                  f"Use the -l/--list option to see all available datasets.",)
        raise e


if __name__ == "__main__":
    main()
