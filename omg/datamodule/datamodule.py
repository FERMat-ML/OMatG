import hashlib
from importlib.resources import files
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Callable

import pickle as pkl
import lmdb
import torch
from loguru import logger
from monty.dev import requires
from tqdm import tqdm


# For type checking
if TYPE_CHECKING:
    from colabfit.tools.configuration import Configuration as ColabfitConfiguration
    from colabfit.tools.database import MongoDatabase

# check if colabfit-tools is installed
try:
    from colabfit.tools.database import MongoDatabase
except ImportError:
    MongoDatabase = None

import ase.io
from ase.data import chemical_symbols

# map from file_format to file extension
SUPPORTED_FORMAT = {"xyz": ".xyz"}
SUPPORTED_PARSERS = ["ase"]


class Configuration:
    r"""
    Class of atomic configuration.
    This is used to store the information of an atomic configuration, e.g. supercell,
    species, coords.

    Args:
        cell: A 3x3 matrix of the lattice vectors. The first, second, and third rows are
            :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively.
        species: A list of N strings giving the species of the atoms, where N is the
            number of atoms.
        coords: A Nx3 matrix of the coordinates of the atoms, where N is the number of
            atoms.
        PBC: A list with 3 components indicating whether periodic boundary condition
            is used along the directions of the first, second, and third lattice vectors.
        identifier: a (unique) identifier of the configuration
    """

    def __init__(
        self,
        cell: torch.Tensor,
        species: List[str],
        coords: torch.Tensor,
        PBC: List[bool],
        identifier: Optional[Union[str, Path]] = None,
        property_dict: Optional[Dict] = {},
    ):
        self._cell = cell
        self._species = species
        self._coords = coords
        self._PBC = PBC
        self._fingerprint = None

        self._identifier = identifier
        self._path = None

        self._property_dict = property_dict
        self._metadata: dict = {}
        # TODO: Dynamic loading from colabfit-tools dataset. Is it needed?

    @classmethod
    def from_colabfit(
        cls,
        database_client: "MongoDatabase",
        data_object: dict,
    ):
        """
        Read configuration from colabfit database .

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
                fetch database from colabfit-tools dataset.
            data_object: colabfit data object dictionary to be associated with current
                configuration and property.
            weight: an instance that computes the weight of the configuration in the loss
                function.
        """
        try:
            configuration_id = data_object["relationships"][0]["configuration"]
            fetched_configuration = database_client.get_cleaned_configuration(
                configuration_id
            )
            fetched_properties = list(
                database_client.get_cleaned_property_instances(
                    data_object["relationships"][0]["property_instance"]
                )
            )
        except:
            raise ConfigurationError(
                "Looks like Mongo database did not return appropriate response. "
                f"Please run db.configurations.find('_id':{data_object}) to verify response. "
            )
        cell = torch.asarray(fetched_configuration["cell"])
        # TODO: consistent Z -> symbol mapping -> Z mapping across all kliff
        species = [
            chemical_symbols[int(i)] for i in fetched_configuration["atomic_numbers"]
        ]
        coords = torch.asarray(fetched_configuration["positions"])
        PBC = [bool(i) for i in fetched_configuration["pbc"]]

        self = cls(
            cell,
            species,
            coords,
            PBC,
            identifier=configuration_id,
        )
        self.metadata = {
            "do-id": data_object["colabfit-id"],
            "co-id": fetched_configuration["colabfit-id"],
            "pi-ids": [pi["colabfit-id"] for pi in fetched_properties],
            "names": fetched_configuration["names"],
        }
        # Update self.metadata with information from metadata collection
        md_dict = database_client.get_metadata_from_do_doc(data_object)
        if md_dict:
            md_dict["md-id"] = md_dict["colabfit-id"]
            md_dict.pop("colabfit-id")
            self.metadata.update(md_dict)

        return self

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: ase.Atoms,
    ):
        """
        Read configuration from ase.Atoms object.

        Args:
            atoms: ase.Atoms object.
        """
        cell = atoms.get_cell().complete()[:]
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        PBC = atoms.get_pbc()

        self = cls(
            cell,
            species,
            coords,
            PBC,
        )
        return self

    def to_ase_atoms(self):
        """
        Convert the configuration to ase.Atoms object.

        Returns:
            ase.Atoms representation of the Configuration
        """
        atoms = ase.Atoms(
            symbols=self.species,
            positions=self.coords,
            cell=self.cell,
            pbc=self.PBC,
        )
        return atoms

    @classmethod
    def from_lmdb(
        cls,
        env: lmdb.Environment,
        key: str,
        dynamic: bool = False,
        property_keys: Tuple[str] = None,
        floating_point_precision: torch.dtype = torch.float64
    ):
        """
        Read configuration from lmdb.

        Args:
            env: lmdb.Environment object.
            key: key to read the configuration from the lmdb.
        """
        if not dynamic:
            with (env.begin() as txn):
                data = txn.get(key.encode())
                if data is None:
                    raise ConfigurationError(f"Key {key} not found in the lmdb.")
                lmdb_config = pkl.loads(data)
                cell = lmdb_config.get("cell", None)
                if cell is not None:
                    cell = cell.to(floating_point_precision)
                species = lmdb_config.get("atomic_numbers", None)
                coords = lmdb_config.get("pos", None)
                if coords is not None:
                    coords = coords.to(floating_point_precision)
                PBC = lmdb_config.get("pbc", None)
                ds_idx = lmdb_config.get("ds_idx", 0)
                identifier = key

                if property_keys:
                    property_dict = {}
                    if isinstance(property_keys, str):
                        property_keys = (property_keys,)
                    for prop in property_keys:
                        property_dict[prop] = lmdb_config.get(prop, None)
                        if (property_dict[prop] is not None and torch.is_tensor(property_dict[prop])
                                and property_dict[prop].is_floating_point()):
                            property_dict[prop] = property_dict[prop].to(floating_point_precision)
                else:
                    property_dict = {}

                species = [chemical_symbols[int(i)] for i in species]
                PBC = [bool(i) for i in PBC]
                config = cls(
                    cell,
                    species,
                    coords,
                    PBC,
                    identifier=identifier,
                    property_dict=property_dict,
                )
                config.metadata |= {"ds_idx": ds_idx}
            return config
        else:
            config = Configuration(None, None, None, None, identifier=key)
            config.metadata = {"lmdb-env": env}

    @property
    def cell(self) -> torch.Tensor:
        """
        3x3 matrix of the lattice vectors of the configurations.
        """
        return self._cell

    @property
    def PBC(self) -> List[bool]:
        """
        A list with 3 components indicating whether periodic boundary condition
        is used along the directions of the first, second, and third lattice vectors.
        """
        return self._PBC

    @property
    def species(self) -> List[str]:
        """
        Species string of all atoms.
        """
        return self._species

    @property
    def coords(self) -> torch.Tensor:
        """
        A Nx3 matrix of the Cartesian coordinates of all atoms.
        """
        return self._coords

    @property
    def identifier(self) -> str:
        """
        Return identifier of the configuration.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        """
        Set the identifier of the configuration.
        """
        self._identifier = identifier

    @property
    def fingerprint(self):
        """
        Return the stored fingerprint of the configuration.
        """
        return self._fingerprint

    @fingerprint.setter
    def fingerprint(self, fingerprint):
        """
        Set the fingerprint of the configuration.
        Args:
         fingerprint: Object which is the fingerprint of the configuration.
        """
        self._fingerprint = fingerprint

    @property
    def path(self) -> Union[Path, None]:
        """
        Return the path of the file containing the configuration. If the configuration
        is not read from a file, return None.
        """
        return self._path

    @property
    def metadata(self) -> dict:
        """
        Return the metadata of the configuration.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """
        Set the metadata of the configuration.
        """
        self._metadata = metadata

    @property
    def property_dict(self) -> dict:
        """
        Return the property dictionary of the configuration.
        """
        return self._property_dict

    @property_dict.setter
    def property_dict(self, property_dict: dict):
        """
        Set the property dictionary of the configuration.
        """
        self._property_dict = property_dict

    def get_num_atoms(self) -> int:
        """
        Return the total number of atoms in the configuration.
        """
        return len(self.species)

    def get_num_atoms_by_species(self) -> Dict[str, int]:
        """
        Return a dictionary of the number of atoms with each species.
        """
        return self.count_atoms_by_species()

    def get_volume(self) -> float:
        """
        Return volume of the configuration.
        """
        return abs(torch.dot(torch.cross(self.cell[0], self.cell[1]), self.cell[2]))

    def count_atoms_by_species(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Count the number of atoms by species.

        Args:
            symbols: species to count the occurrence. If `None`, all species present
                in the configuration are used.

        Returns:
            {specie, count}: with `key` the species string, and `value` the number of
                atoms with each species.
        """

        unique, counts = torch.unique(self.species, return_counts=True)
        symbols = unique if symbols is None else symbols

        natoms_by_species = dict()
        for s in symbols:
            if s in unique:
                natoms_by_species[s] = counts[list(unique).index(s)]
            else:
                natoms_by_species[s] = 0

        return natoms_by_species

    def order_by_species(self):
        """
        Order the atoms according to the species such that atoms with the same species
        have contiguous indices.
        """
        species, coords = zip(
                *sorted(zip(self.species, self.coords), key=lambda pair: pair[0])
        )
        self._species = torch.asarray(species)
        self._coords = torch.asarray(coords)

    @staticmethod
    def _get_colabfit_property(
        database_client: "MongoDatabase",
        property_id: Union[List[str], str],
        property_name: str,
        property_type: str,
    ):
        """
        Returns colabfit-property. workaround till we get proper working get_property routine

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
                fetch database from colabfit-tools dataset.
            property_id: colabfit ID of the property instance to be associated with
                current configuration.
            property_name: subfield of the property to fetch
            property_type: type of property to fetch

        Returns:
            Property: fetched value, None if query comes empty
        """
        pi_doc = database_client.property_instances.find_one(
            {"colabfit-id": {"$in": property_id}, "type": property_type}
        )
        if pi_doc:
            return pi_doc[property_type][property_name]["source-value"]
        else:
            return None

    def to_dict(self):
        """
        Return a dictionary representation of the configuration.
        Returns:
            A dictionary representation of the configuration.
        """
        if self.coords is not None:
            config_dict = {
                "cell": self.cell,
                "species": self.species,
                "coords": self.coords,
                "PBC": self.PBC,
                "identifier": self.identifier,
            }
        else:  # it might be dynamic lmdb
            if self.metadata.get("lmdb-env"):
                try:
                    data = (
                        self.metadata["lmdb-env"].begin().get(self.identifier.encode())
                    )
                    lmdb_config = pkl.loads(data)
                    config_dict = {
                        "cell": lmdb_config.get("cell", None),
                        "species": [
                            chemical_symbols[int(i)]
                            for i in lmdb_config.get("atomic_numbers", None)
                        ],
                        "coords": lmdb_config.get("pos", None),
                        "PBC": lmdb_config.get("pbc", None),
                        "identifier": self.identifier,
                    }
                except Exception as e:
                    raise ConfigurationError(f"Error reading lmdb: {e}")
            else:
                raise ConfigurationError("Configuration does not contain coordinates.")
        return config_dict


class DataModule:
    """
    A dataset of multiple configurations (:class:`~kliff.dataset.Configuration`).

    Args:
        configurations: A list of :class:`~kliff.dataset.Configuration` objects.
    """

    def __init__(self, lmdb_paths=None, property_keys=None, trainer_precision: Union[int, str, None] = "64-true"):
        # if configurations is None:
        #     self._configs = []
        # elif isinstance(configurations, Iterable) and not isinstance(
        #     configurations, str
        # ):
        #     self._configs = list(configurations)
        # else:
        #     raise DataModuleError(
        #         "configurations must be a iterable of Configuration objects."
        #     )
        if trainer_precision == "64-true" or trainer_precision == "64" or trainer_precision == 64:
            self._floating_point_precision = torch.float64
        elif (trainer_precision is None or trainer_precision == "32-true" or trainer_precision == "32" or
              trainer_precision == 32 or trainer_precision == "16-mixed" or trainer_precision == "bf16-mixed"):
            self._floating_point_precision = torch.float32
        elif (trainer_precision == "16-true" or trainer_precision == "16" or trainer_precision == 16
              or trainer_precision == "transformer-engine-float16"):
            self._floating_point_precision = torch.float16
        elif (trainer_precision == "bf16-true" or trainer_precision == "bf16"
              or trainer_precision == "transformer-engine"):
            self._floating_point_precision = torch.bfloat16
        else:
            raise ValueError(f"Unknown trainer precision: {trainer_precision}")

        self._metadata: dict = {}
        self._return_config_on_getitem = True
        self._property_keys = None

        cache_dir = Path("./cache")
        os.makedirs(cache_dir, exist_ok=True)

        if lmdb_paths is not None:
            try:
                self.from_lmdb(lmdb_paths, property_keys=property_keys, save_path=cache_dir)
            except lmdb.Error:
                # Try to use the data from the omg package.
                if isinstance(lmdb_paths, Iterable):
                    package_lmdb_paths = [files("omg").joinpath(lmdb_path) for lmdb_path in lmdb_paths]
                    self.from_lmdb(package_lmdb_paths, property_keys=property_keys, save_path=cache_dir)
                else:
                    self.from_lmdb(files("omg").joinpath(lmdb_paths), property_keys=property_keys, save_path=cache_dir)

    @classmethod
    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def from_colabfit(
        cls,
        colabfit_database: str,
        colabfit_dataset: str,
        colabfit_uri: str = "mongodb://localhost:27017",
        **kwargs,
    ) -> "DataModule":
        """
        Read configurations from colabfit database and initialize a dataset.

        Args:
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            colabfit_database: Name of the colabfit Mongo database to read from.
            colabfit_dataset: Name of the colabfit dataset instance to read from, usually
                it is of form, e.g., "DS_xxxxxxxxxxxx_0"
            colabfit_uri: connection URI of the colabfit Mongo database to read from.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_colabfit(
            colabfit_database, colabfit_dataset, colabfit_uri, **kwargs
        )
        return instance

    @staticmethod
    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def _read_from_colabfit(
        database_client: MongoDatabase,
        colabfit_dataset: str,
    ) -> List[Configuration]:
        """
        Read configurations from colabfit database.

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
                fetch database from colabfit-tools dataset.
            colabfit_dataset: Name of the colabfit dataset instance to read from.
            weight: an instance that computes the weight of the configuration in the loss
                function.

        Returns:
            A list of configurations.
        """
        # get configuration and property ID and send it to load configuration-first get Data Objects
        data_objects = database_client.data_objects.find(
            {"relationships.dataset": colabfit_dataset}
        )
        if not data_objects:
            logger.error(f"{colabfit_dataset} is either empty or does not exist")
            raise DataModuleError(f"{colabfit_dataset} is either empty or does not exist")

        configs = []
        for data_object in data_objects:
            configs.append(
                Configuration.from_colabfit(database_client, data_object)
            )

        if len(configs) <= 0:
            raise DataModuleError(f"No dataset file with in {colabfit_dataset} dataset.")

        logger.info(f"{len(configs)} configurations read from {colabfit_dataset}")

        return configs

    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def add_from_colabfit(
        self,
        colabfit_database: str,
        colabfit_dataset: str,
        colabfit_uri: str = "mongodb://localhost:27017",
        **kwargs,
    ):
        """
        Read configurations from colabfit database and add them to the dataset.

        Args:
            colabfit_database: Name of the colabfit Mongo database to read from.
            colabfit_dataset: Name of the colabfit dataset instance to read from (usually
                it is of form, e.g., "DS_xxxxxxxxxxxx_0")
            colabfit_uri: connection URI of the colabfit Mongo database to read from.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).

        """
        # open link to the mongo
        mongo_client = MongoDatabase(colabfit_database, uri=colabfit_uri, **kwargs)
        configs = DataModule._read_from_colabfit(mongo_client, colabfit_dataset, None)
        self._configs.extend(configs)

    @classmethod
    def from_ase(
        cls,
        path: Union[Path, str] = None,
        ase_atoms_list: List[ase.Atoms] = None,
        slices: Union[slice, str] = ":",
        file_format: str = "xyz",
    ) -> "DataModule":
        """
        Read configurations from ase.Atoms object and initialize a dataset. The expected
        inputs are either a pre-initialized list of ase.Atoms, or a path from which
        the dataset can be read from (usually an extxyz file). If the configurations
        are in a file, or a directory, it would use ~ase.io.read() to read the
        configurations. Therefore, it is expected that the file format is supported by
        ASE.

        Example:
            >>> from ase.build import bulk
            >>> ase_configs = [bulk("Al"), bulk("Al", cubic=True)]
            >>> dataset_from_list = DataModule.from_ase(ase_atoms_list=ase_configs)
            >>> dataset_from_file = DataModule.from_ase(path="configs.xyz", energy_key="Energy")

        Args:
            path: Path the directory (or filename) storing the configurations.
            ase_atoms_list: A list of ase.Atoms objects.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_ase(
            path, ase_atoms_list, slices, file_format
        )
        return instance

    @staticmethod
    def _read_from_ase(
        path: Path = None,
        ase_atoms_list: List[ase.Atoms] = None,
        slices: str = ":",
        file_format: str = "xyz",
    ) -> List[Configuration]:
        """
        Read configurations from ase.Atoms object. If the configurations are in a file,
        or a directory, it would use ~ase.io.read() to read the configurations.

        Args:
            path: Path the directory (or filename) storing the configurations.
            ase_atoms_list: A list of ase.Atoms objects.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A list of configurations.
        """
        if ase_atoms_list is None and path is None:
            raise DataModuleError(
                "Either list of ase.Atoms objects or a path must be provided."
            )

        if ase_atoms_list:
            configs = [
                Configuration.from_ase_atoms(
                    config,
                )
                for config in ase_atoms_list
            ]
        else:
            try:
                extension = SUPPORTED_FORMAT[file_format]
            except KeyError:
                raise DataModuleError(
                    f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                    f"got: {file_format}."
                )

            path = Path(path)

            if path.is_dir():
                parent = path
                all_files = []
                for root, dirs, files in os.walk(parent, followlinks=True):
                    for f in files:
                        if f.endswith(extension):
                            all_files.append(Path(root).joinpath(f))
                all_files = sorted(all_files)
            else:
                parent = path.parent
                all_files = [path]

            if len(all_files) == 1:  # single xyz file with multiple configs
                all_configs = ase.io.read(all_files[0], index=slices)

                configs = [
                    Configuration.from_ase_atoms(
                        config,
                    )
                    for config in all_configs
                ]
            else:
                configs = [
                    Configuration.from_ase_atoms(
                        ase.io.read(f),
                    )
                    for f in all_files
                ]

        if len(configs) <= 0:
            raise DataModuleError(
                f"No dataset file with file format `{file_format}` found at {path}."
            )

        logger.info(f"{len(configs)} configurations loaded using ASE.")
        return configs

    def add_from_ase(
        self,
        path: Union[Path, str] = None,
        ase_atoms_list: List[ase.Atoms] = None,
        slices: str = ":",
        file_format: str = "xyz",
    ):
        """
        Read configurations from ase.Atoms object and append to a dataset. The expected
        inputs are either a pre-initialized list of ase.Atoms, or a path from which
        the dataset can be read from (usually an extxyz file). If the configurations
        are in a file, or a directory, it would use ~ase.io.read() to read the
        configurations. Therefore, it is expected that the file format is supported by
        ASE.

        Example:
            >>> from ase.build import bulk
            >>> ase_configs = [bulk("Al"), bulk("Al", cubic=True)]
            >>> dataset = DataModule()
            >>> dataset.add_from_ase(ase_atoms_list=ase_configs)
            >>> dataset.add_from_ase(path="configs.xyz", energy_key="Energy")

        Args:
            path: Path the directory (or filename) storing the configurations.
            ase_atoms_list: A list of ase.Atoms objects.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.
        """
        if isinstance(path, str):
            path = Path(path)

        configs = self._read_from_ase(
            path, ase_atoms_list,file_format=file_format, slices=slices
        )
        self._configs.extend(configs)

    # @classmethod
    def from_lmdb(
        self,
        path: Union[Path, str, List[Path], List[str]],
        dynamic_loading: bool = True,
        subdir: bool = False,
        save_path: Optional[Path] = None,
        reuse: bool = True,
        checksum: Optional[str] = None,
        property_keys: Tuple[str] = None,
        use_transformed_if_available: bool = True
    ):
        """
        Read configurations from LMDB file and append to a dataset.

        Args:
            path: Path to the LMDB file.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            sharded: Whether the LMDB file is sharded.
            dynamic_loading: Whether to load the data dynamically.
            subdir: Whether the data is stored in subdirectories.
            length: Number of configurations to load.
        """
        # instance = self()
        self.add_from_lmdb(
            path, dynamic_loading, subdir, save_path, reuse, checksum, property_keys, use_transformed_if_available
        )
        self._property_keys = property_keys
        return self

    def add_from_lmdb(
        self,
        path: Union[Path, str, List[Path], List[str]],
        dynamic_loading: bool = True,
        subdir: bool = False,
        save_path: Optional[Path] = None,
        reuse: bool = True,
        checksum: Optional[str] = None,
        property_keys: Tuple[str] = None,
        use_transformed_if_available: bool = True
    ):
        """
        Read configurations from LMDB file and append to a dataset.

        Args:
            path: Path to the LMDB file(s).
            weight: Instance that computes the configuration weight in the loss function.
            sharded: Whether the LMDB file is sharded.
            dynamic_loading: Whether to load the data dynamically.
            subdir: Whether the data is stored in subdirectories.
            save_path: Path to save the master LMDB file.
            reuse: Whether to reuse existing master LMDB file.
            checksum: Checksum for the dataset.
        """
        if dynamic_loading:
            self._configs = torch.asarray([], dtype=torch.int64)
            self.add_metadata({"dynamic": True})

        path = [path] if isinstance(path, (str, Path)) else path
        save_path = Path("./") if not save_path else save_path

        if not checksum:
            ds_hash = "|".join([str(p) for p in path])
            checksum = hashlib.md5(ds_hash.encode()).hexdigest()
        master_lmdb = save_path.joinpath(f"dataset_{checksum}.lmdb")

        if master_lmdb.exists() and reuse:
            master_lmdb_env = lmdb.open(
                str(master_lmdb),
                subdir=False,
                readonly=True,
                lock=False,
                map_size=int(1e12),
            )
            with master_lmdb_env.begin() as txn:
                n_configs = txn.stat()["entries"]
            self._configs = torch.arange(n_configs, dtype=torch.int64)
            logger.info(f"Reusing existing LMDB file: {master_lmdb}")
            process_configs = False
        else:
            if master_lmdb.exists():
                os.remove(master_lmdb)
            master_lmdb_env = lmdb.open(
                str(master_lmdb),
                subdir=False,
                readonly=False,
                lock=False,
                map_size=int(1e12),
            )
            logger.info(f"Creating new LMDB file: {master_lmdb}")
            process_configs = True

        self.add_metadata({"master_lmdb": master_lmdb, "master_env": master_lmdb_env})

        for lmdb_idx, lmdb_path in enumerate(path):
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=subdir)
            configs = self._read_from_lmdb(env,  dynamic_loading, subdir, property_keys, self._floating_point_precision)

            if dynamic_loading:
                self.metadata.setdefault("lmdb_envs", []).append(env)
                if process_configs:
                    configs_idx = torch.arange(
                        len(self._configs), len(self._configs) + len(configs), dtype=torch.int64
                    )
                    self._configs = torch.cat((self._configs, configs_idx), dim=0)
                    with self.metadata["master_env"].begin(write=True) as txn:

                        print(f"Adding {len(configs)} configurations to LMDB from {lmdb_path}")
                        pbar = tqdm(total=len(configs))
                        for i, keys in zip(configs_idx, configs):
                            i_ = i.item() if isinstance(i, torch.Tensor) else i
                            idx = {"lmdb_env": f"{lmdb_idx}", "config_key": f"{keys}"}
                            txn.put(str(i_).encode(), pkl.dumps(idx))
                            pbar.update(1)
            else:
                self._configs.extend(configs)
                env.close()

    @staticmethod
    def _read_from_lmdb(
        env: lmdb.Environment,
        dynamic_loading: bool,
        subdir: bool,
        property_keys: Tuple[str] = None,
        floating_point_precision: torch.dtype = torch.float64
    ) -> Union[List[Configuration], List[str]]:
        """
        Read configurations from LMDB file.

        Args:
            path:
            weight:
            sharded:
            dynamic_loading:
            subdir:
            length:

        Returns:

        """
        with env.begin() as txn:
            num_keys = txn.stat()["entries"]
            keys = [key.decode() for key, _ in txn.cursor()]

        if not dynamic_loading:
            configs = [
                Configuration.from_lmdb(env, key, dynamic=False, property_keys=property_keys,
                                        floating_point_precision=floating_point_precision) for key in keys
            ]
        else:
            configs = keys
        return configs

    @property
    def configs(self) -> List[Configuration]:
        """
        Return the configurations in the dataset.
        """
        if not self._configs:
            raise DataModuleError(
                "No configurations found, maybe you are using dynamic loading?"
            )
        return self._configs

    def get_configs(self) -> List[Configuration]:
        """
        Get shallow copy of the configurations.
        """
        return self._configs[:]

    def __len__(self) -> int:
        """
        Get length of the dataset. It is needed to make dataset directly compatible
        with various dataloaders.

        Returns:
            Number of configurations in the dataset.
        """
        return len(self._configs)

    def __getitem__(
        self, idx: Union[int, torch.Tensor, List]
    ) -> Union[Configuration, "DataModule"]:
        """
        Get the configuration at index `idx`. If the index is a list, it returns a new
        dataset with the configurations at the indices.

        Args:
         idx: Index of the configuration to get or a list of indices.

        Returns:
            The configuration at index `idx` or a new dataset with the configurations at
            the indices.
        """
        if self._metadata.get("dynamic"):
            if isinstance(idx, int):
                with self.metadata["master_env"].begin() as txn:
                    lmdb_idx = txn.get(f"{idx}".encode())
                    lmdb_idx = pkl.loads(lmdb_idx)

                key = lmdb_idx["config_key"]
                idx = int(lmdb_idx["lmdb_env"])
                env = self.metadata["lmdb_envs"][idx]
                return Configuration.from_lmdb(env, key, property_keys=self._property_keys,
                                               floating_point_precision=self._floating_point_precision)

            else:
                if not self._return_config_on_getitem:
                    ds = DataModule()
                    ds._metadata = self._metadata
                    ds._configs = idx
                    return ds
                else:
                    configs = []
                    for i in idx:
                        with self.metadata["master_env"].begin() as txn:
                            lmdb_idx = txn.get(f"{i}".encode())
                            lmdb_idx = pkl.loads(lmdb_idx)
                        key = lmdb_idx["config_key"]
                        idx = int(lmdb_idx["lmdb_env"])
                        env = self.metadata["lmdb_envs"][idx]
                        configs.append(Configuration.from_lmdb(env, key, property_keys=self._property_keys,
                                                               floating_point_precision=self._floating_point_precision))
                    return DataModule(configs)
        else:
            if isinstance(idx, int):
                return self._configs[idx]
            else:
                return DataModule([self._configs[i] for i in idx])

    def add_metadata(self, metadata: dict):
        """
        Add metadata to the dataset object.

        Args:
            metadata: A dictionary containing the metadata.
        """
        if not isinstance(metadata, dict):
            raise DataModuleError("metadata must be a dictionary.")
        self._metadata |= metadata

    def get_metadata(self, key: str):
        """
        Get the metadata of the dataset.

        Args:
            key: Key of the metadata to get.

        Returns:
            Value of the metadata.
        """
        return self._metadata[key]

    @property
    def metadata(self):
        """
        Return the metadata of the dataset.
        """
        return self._metadata

    @staticmethod
    def get_manifest_checksum(
        dataset_manifest: dict, transform_manifest: Optional[dict] = None
    ) -> str:
        """
        Get the checksum of the dataset manifest.

        Args:
            dataset_manifest: Manifest of the dataset.
            transform_manifest: Manifest of the transformation.

        Returns:
            Checksum of the manifest.
        """
        dataset_str = json.dumps(dataset_manifest, sort_keys=True)
        if transform_manifest:
            transform_str = json.dumps(transform_manifest, sort_keys=True)
            dataset_str += transform_str
        return hashlib.md5(dataset_str.encode()).hexdigest()

    @staticmethod
    def get_dataset_from_manifest(dataset_manifest: dict) -> "DataModule":
        """
        Get a dataset from a manifest.

        Examples:
           1.  Manifest file for initializing dataset using ASE parser:
            ```yaml
            dataset:
                type: ase           # ase or path or colabfit
                path: Si.xyz        # Path to the dataset
                save: True          # Save processed dataset to a file
                save_path: /folder/to   # Save to this folder
                shuffle: False      # Shuffle the dataset
                weights: /path/to/weights.dat # or dictionary with weights
                keys:
                    energy: Energy  # Key for energy, if ase dataset is used
                    forces: forces  # Key for forces, if ase dataset is used
            ```

            2. Manifest file for initializing dataset using KLIFF extxyz parser:
            ```yaml
            dataset:
                type: path          # ase or path or colabfit
                path: /all/my/xyz   # Path to the dataset
                save: False         # Save processed dataset to a file
                shuffle: False      # Shuffle the dataset
                weights:            # same weight for all, or file with weights
                    config: 1.0
                    energy: 0.0
                    forces: 10.0
                    stress: 0.0
            ```

            3. Manifest file for initializing dataset using ColabFit parser:
            ```yaml
            dataset:
                type: colabfit      # ase or path or colabfit
                save: False         # Save processed dataset to a file
                shuffle: False      # Shuffle the dataset
                weights: None
                colabfit_dataset:
                    dataset_name:
                    database_name:
                    database_url:

            4. Manifest file for initializing dataset using LMDB:
            ```yaml
            dataset:
                type: lmdb          # ase or path or colabfit
                lmbd_paths:         # Path to the lmdb file
                    - /path/to/lmdb
                    - /path/to/lmdb
                dynamic_loading: True
                save: True         # Save processed dataset to a file, or reuse
                path: /path/to/save # Save to this folder
            ```


        Args:
            dataset_manifest: List of configurations.
            save_path: Path to save the dataset.

        Returns:
            A dataset of configurations.
        """
        dataset_type = dataset_manifest.get("type").lower()
        if (
            dataset_type != "ase"
            and dataset_type != "colabfit"
            and dataset_type != "lmdb"
        ):
            raise DataModuleError(f"Dataset type {dataset_type} not supported.")

        if dataset_type == "ase":
            dataset = DataModule.from_ase(
                path=dataset_manifest.get("path", "."),
            )
        elif dataset_type == "colabfit":
            try:
                colabfit_dataset = dataset_manifest.get("colabfit_dataset")
                colabfit_database = colabfit_dataset.database_name
            except KeyError:
                raise DataModuleError("Colabfit dataset or database not provided.")
            colabfit_uri = dataset_manifest.get(
                "colabfit_uri", "mongodb://localhost:27017"
            )

            dataset = DataModule.from_colabfit(
                colabfit_database=colabfit_database,
                colabfit_dataset=colabfit_dataset,
                colabfit_uri=colabfit_uri,
            )
        elif dataset_type == "lmdb":
            dataset = DataModule.from_lmdb(
                path=dataset_manifest.get("lmdb_paths", []),
                dynamic_loading=dataset_manifest.get("dynamic_loading", True),
                save_path=Path(dataset_manifest.get("path", "./")),
                reuse=dataset_manifest.get("save", True),
                checksum=DataModule.get_manifest_checksum(dataset_manifest),
            )
        else:
            # this should not happen
            raise DataModuleError(f"Dataset type {dataset_type} not supported.")

        return dataset

    # __del__ closes te lmdb connections in most inopportune times
    # not using it till I figure out how to handle python garbage collection properly
    # def __del__(self):
    #     self.cleanup(save=False)

    def cleanup(self, save: bool = False):
        if self.metadata.get("lmdb_envs"):
            for env in self.metadata["lmdb_envs"]:
                env.close()

        if self.metadata.get("master_env"):
            self.metadata["master_env"].close()

        if not save:
            if self.metadata.get("master_lmdb"):
                logger.info(f"Removing master LMDB file: {self.metadata['master_lmdb']}")
                shutil.rmtree(self.metadata["master_lmdb"], ignore_errors=True)
                # if lmdb file is a file, not directory, remove the file
                if self.metadata["master_lmdb"].is_file():
                    self.metadata["master_lmdb"].unlink()



    def toggle_lazy_config_fetch(self):
        """
        Toggle the lazy configuration fetch. If enabled, the configurations are not
        loaded when the dataset is initialized, but are loaded when they are accessed.
        """
        # self._return_config_on_getitem = not self._return_config_on_getitem
        # if not self._return_config_on_getitem:
        #     logger.warning("Lazy config fetch for seq: Enabled")
        # else:
        #     logger.warning(f"Lazy config fetch for seq: Disabled")
        return self

    def __enter__(self):
        self._return_config_on_getitem = False
        logger.warning("Lazy config fetch for seq: Enabled")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._return_config_on_getitem = True
        logger.warning(f"Lazy config fetch for seq: Disabled")

    def transform_and_save(self, transform: Union[List[Callable], Callable], attributes: List, save_path: Path):
        """
        Apply a transformation to the dataset and save the transformed dataset.

        Args:
            transform: Transformation to apply to the dataset.
            save_path: Path to save the transformed dataset.
        """

    pass


class OverfittingDataModule(DataModule):
    """
    Datamodule that always returns a single configuration.

    Can be used to overfit the model to a single configuration.
    """
    def __init__(self, lmdb_paths=None, property_keys=None, structure_index: int = 0) -> None:
        super().__init__(lmdb_paths=lmdb_paths, property_keys=property_keys)
        if not 0 <= structure_index < len(self):
            raise DataModuleError(f"Invalid structure index {structure_index}, "
                                      f"possible values are 0 to {len(self) - 1}.")
        self._structure_index = structure_index

    def __getitem__(self, idx: Union[int, torch.Tensor, List]) -> Union[Configuration, "DataModule"]:
        """
        Get the configuration at index `idx`. If the index is a list, it returns a new
        dataset with the configurations at the indices.

        This method ignores the given indices and always return the same configuration.

        Args:
         idx: Index of the configuration to get or a list of indices.

        Returns:
            The fixed configuration at a fixed index or a new dataset with the the same configuration replicated.
        """
        if isinstance(idx, int):
            return super().__getitem__(self._structure_index)
        else:
            return super().__getitem__([self._structure_index for _ in idx])


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg


class DataModuleError(Exception):
    def __init__(self, msg):
        super(DataModuleError, self).__init__(msg)
        self.msg = msg
