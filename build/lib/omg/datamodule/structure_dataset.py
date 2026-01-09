import hashlib
from importlib.resources import files
from pathlib import Path
import pickle
from typing import Any, Optional, Sequence, Union
import warnings
from ase.io import write
import lmdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure as PymatgenStructure, Lattice as PymatgenLattice
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from omg.datamodule.structure import Structure


class StructureDataset(Dataset):
    """
    Dataset for reading crystalline structures from several file formats.

    This dataset optionally allows for lazy reading of the structures. In this case, it reads the structures lazily from
    a file when they are requested. This is achieved with the help of the LMDB database format, which allows for
    efficient key-value storage and retrieval.

    If lazy_storage is set to False, all structures are read from the file during initialization and stored in memory.
    If lazy_storage is set to True, only the structure keys in the LMDB database are stored in memory, and the
    structures are read from the LMDB file when they are requested via the __getitem__ method.

    The supported file formats for reading crystalline structures are LMDB, CSV, and Parquet. The specific format is
    inferred from the file extension.

    The _from_lmdb, _from_csv, and _from_parquet methods are used to read the structures from the respective file
    formats on initialization of this class. If lazy_storage is set to True, the _from_lmdb method also serves as a
    test case for the LMDB reading functionality.

    Since preprocessing CSV files can be costly, the _from_csv method reads structures from a CSV file containing CIF
    strings and converts them to the LMDB format for efficient future access. The created LMDB cache file will also be
    used for reading structures lazily if lazy_storage is set to True. The LMDB cache file is created in the same
    directory as the CSV file and is named based on the hash of the CSV file and the preprocessing options used. If the
    cache file already exists, it is used directly instead of creating a new one.

    Likewise, since preprocessing Parquet files can be costly, the _from_parquet method reads structures from a Parquet
    file and converts them to the LMDB format for efficient future access and lazy storage.

    The _from_lmdb, _from_csv, and _from_parquet methods document the expected file contents for their respective file
    types in more detail.

    This class also provides methods to save all the structures in LMDB or xyz format.

    The returned structures can optionally be converted to fractional coordinates and Niggli reduced.

    :param file_path:
        Path to the file containing the structures.
        This can either be an absolute path, a relative path to the current working directory, or a relative path within
        the omg package.
        Supported formats are .lmdb, .csv, and .parquet.
    :type file_path: str
    :param property_keys:
        An optional sequence of property keys that should be read from the file and stored within the structures.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param floating_point_precision:
        Floating point precision to use for the structures. Can be one of "64-true", "64", "32-true", "32",
        "16-true", "16", "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        Defaults to "64-true".
    :type floating_point_precision: Union[int, str, None]
    :param lazy_storage:
        Whether to read the structures lazily from a LMDB file when they are requested.
        Defaults to True.
    :type lazy_storage: bool
    :param convert_to_fractional:
        Whether to convert the atomic positions to fractional coordinates in the returned structures.
        Defaults to True.
    :type convert_to_fractional: bool
    :param niggli_reduce:
        Whether to apply a Niggli reduction to the returned structures.
        Defaults to False.
    :type niggli_reduce: bool
    :param cdvae_preprocessing:
        Whether to preprocess the structures to match the conventions used in CDVAE when reading from CSV files.
        When reading from LMDB files, this option is ignored.
        This applies the Niggli reduction of PyMatgen.
        Defaults to None.
    :type cdvae_preprocessing: Optional[bool]
    :param mattergen_preprocessing:
        Whether to preprocess the structures to match the conventions used in MatterGen when reading from CSV files.
        When reading from LMDB files, this option is ignored.
        This applies the primitive cell extraction followed by the Niggli reduction of PyMatgen.
        Defaults to None.
    :type mattergen_preprocessing: Optional[bool]

    :raises KeyError:
        If any of the property keys are not found in the structure properties.
        If reserved property keys are used (i.e., "cell", "pos", "atomic_numbers", "file_path", "file_key", or
        "identifier").
    """

    def __init__(self, file_path: str, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true", lazy_storage: bool = True,
                 convert_to_fractional: bool = True, niggli_reduce: bool = False,
                 cdvae_preprocessing: Optional[bool] = None, mattergen_preprocessing: Optional[bool] = None) -> None:
        """Constructor for the StructureDataset class."""
        super().__init__()
        self._torch_precision = self._get_torch_precision(floating_point_precision)
        self._property_keys = list(property_keys) if property_keys is not None else []
        for prop in self._property_keys:
            if prop in ("cell", "pos", "atomic_numbers", "file_path", "file_key", "identifier", "ids"):
                raise KeyError(f"Property key '{prop}' is reserved and cannot be used as a property key.")
        path = Path(file_path)

        if path.exists():
            existing_file_path = file_path
        else:
            # Try to use the data from the omg package.
            # noinspection PyTypeChecker
            existing_file_path = Path(files("omg").joinpath(file_path))
            if not existing_file_path.exists():
                raise FileNotFoundError(f"File path {file_path} neither exists on its own or within the omg package.")

        if path.suffix == ".lmdb":
            if cdvae_preprocessing is not None or mattergen_preprocessing is not None:
                warnings.warn("CDVAE and MatterGen preprocessing options are ignored when reading from LMDB files.")
            self._file_path, self._structures = self._from_lmdb(
                existing_file_path, self._property_keys, return_structures=not lazy_storage)
        elif path.suffix == ".parquet":
            if cdvae_preprocessing is not None or mattergen_preprocessing is not None:
                warnings.warn("CDVAE and MatterGen preprocessing options are ignored when reading from LMDB files.")
            self._file_path, self._structures = self._from_parquet(
                existing_file_path, self._property_keys, return_structures=not lazy_storage)
        elif path.suffix == ".csv":
            if cdvae_preprocessing is None or mattergen_preprocessing is None:
                raise ValueError("When reading from CSV files, both cdvae_preprocessing and mattergen_preprocessing "
                                 "options must be specified.")
            self._file_path, self._structures = self._from_csv(
                existing_file_path, self._property_keys, return_structures=not lazy_storage,
                cdvae_preprocessing=cdvae_preprocessing, mattergen_preprocessing=mattergen_preprocessing)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats are .lmdb and .csv.")
        self._number_structures = len(self._structures)
        assert self._number_structures > 0

        self._lazy_storage = lazy_storage
        self._convert_to_fractional = convert_to_fractional
        self._niggli_reduce = niggli_reduce

        # Preprocess structures if not using lazy storage.
        # If using lazy storage, preprocessing will be done when the structures are read.
        if not self._lazy_storage:
            for structure in tqdm(self._structures, desc="Processing structures", unit=" structures"):
                if self._niggli_reduce:
                    structure.niggli_reduce()
                if self._convert_to_fractional:
                    structure.convert_to_fractional()
                structure.to(self._torch_precision)

        if self._lazy_storage:
            # Read structures lazily from this file.
            self._env = lmdb.Environment(str(self._file_path), subdir=False, readonly=True, lock=False, readahead=False,
                                         meminit=False)
        else:
            self._env = None

    @staticmethod
    def _get_torch_precision(floating_point_precision: Union[int, str, None] = "64-true") -> torch.dtype:
        """
        Get the torch precision based on the given floating point precision.

        :param floating_point_precision:
            Floating point precision to use. Can be one of "64-true", "64", "32-true", "32", "16-true", "16",
            "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        :type floating_point_precision: Union[int, str, None]

        :return:
            The corresponding torch dtype.
        :rtype: torch.dtype

        :raises ValueError:
            If the given floating point precision is not recognized.
        """
        if floating_point_precision == "64-true" or floating_point_precision == "64" or floating_point_precision == 64:
            return torch.float64
        elif (floating_point_precision is None or floating_point_precision == "32-true" or
              floating_point_precision == "32" or floating_point_precision == 32 or
              floating_point_precision == "16-mixed" or floating_point_precision == "bf16-mixed"):
            return torch.float32
        elif (floating_point_precision == "16-true" or floating_point_precision == "16" or
              floating_point_precision == 16 or floating_point_precision == "transformer-engine-float16"):
            return torch.float16
        elif (floating_point_precision == "bf16-true" or floating_point_precision == "bf16" or
              floating_point_precision == "transformer-engine"):
            return torch.bfloat16
        else:
            raise ValueError(f"Unknown floating point precision: {floating_point_precision}")

    @staticmethod
    def _compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """
        Compute the hash of a file using the specified algorithm.

        Possible algorithms are those supported by hashlib.file_digest, i.e., typically "md5", "sha1", "sha224",
        "sha256", "sha384", "sha512", "sha3_224", "sha3_256", "sha3_384", "sha3_512", "shake_128", "shake_256",
        "blake2b", and "blake2s".

        :param file_path:
            Path to the file.
        :type file_path: Path
        :param algorithm:
            Hashing algorithm to use. Defaults to "sha256".
        :type algorithm: str

        :return:
            The hexadecimal digest of the file hash.
        :rtype: str

        :raises ValueError:
            If the specified algorithm is not supported.
        """
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hashing algorithm: {algorithm}. "
                             f"Supported algorithms are: {hashlib.algorithms_available}.")
        with open(file_path, "rb") as f:
            digest = hashlib.file_digest(f, algorithm)
        return digest.hexdigest()

    @staticmethod
    def _from_lmdb(existing_lmdb_path: Path, property_keys: Sequence[str],
                   return_structures: bool) -> tuple[Path, Union[list[Structure], list[bytes]]]:
        """
        Check proper format of the LMDB file for crystalline structures, and return its path and the structures or
        their keys within the LMDB file.

        The keys in the LMDB file are expected to be bytes.

        Each key (record) in the LMDB file is expected to contain a pickled dictionary with at least the following keys
        to represent a structure with N atoms:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of Cartesian atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the LMDB file from which the structure was loaded.
        - "file_key": The key in the LMDB file corresponding to the structure.
        - "identifier": If "ids" or "identifier" is present in the LMDB record, it will be stored here. This is usually
                        a unique identifier of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the LMDB file into the structures by specifying them in the
        property_keys parameter. These keys will be read from the LMDB record and stored in the property_dict of the
        structure.

        In order to verify the correctness of the LMDB file, this method reads all records and checks for the presence
        and correct types of the required keys. If return_structures is set to True, the created structures are
        returned. If return_structures is set to False, only the keys of the structures in the LMDB file are returned.

        :param existing_lmdb_path:
            Path to the LMDB file.
        :type existing_lmdb_path: Path
        :param property_keys:
            A sequence of property keys that are read from the LMDB file and stored within the structures.
        :type property_keys: Sequence[str]
        :param return_structures:
            Whether to return the structures read from the LMDB file.
        :type return_structures: bool

        :return:
            The path of the LMDB file.
            The structures in the LMDB file if return_structures is True, otherwise the keys of the structures.
        :rtype: tuple[Path, Union[list[Structure], list[bytes]]]

        :raises TypeError:
            If any of the required keys have incorrect types in the LMDB records.
        """
        structures = []
        with (lmdb.Environment(str(existing_lmdb_path), subdir=False, readonly=True, lock=False) as env,
              env.begin() as txn):
            number_structures = txn.stat()["entries"]

            for enc_key, data in tqdm(txn.cursor(), desc=f"Loading {existing_lmdb_path} data", unit=" structures",
                                      total=number_structures):
                key = enc_key.decode()
                lmdb_structure = pickle.loads(data)

                if "cell" not in lmdb_structure:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'cell'.")
                if not isinstance(lmdb_structure["cell"], torch.Tensor):
                    raise TypeError(f"Key {key} in the lmdb file has 'cell' of type "
                                    f"{type(lmdb_structure['cell'])}, expected torch.Tensor.")
                if not torch.is_floating_point(lmdb_structure["cell"]):
                    raise TypeError(f"Key {key} in the lmdb file has 'cell' of "
                                    f"dtype {lmdb_structure['cell'].dtype}, expected floating point dtype.")

                if "atomic_numbers" not in lmdb_structure:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'atomic_numbers'.")
                if not isinstance(lmdb_structure["atomic_numbers"], torch.Tensor):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' of type "
                                    f"{type(lmdb_structure['atomic_numbers'])}, expected torch.Tensor.")
                if not lmdb_structure["atomic_numbers"].dtype in (torch.int64, torch.int32):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' of dtype "
                                    f"{lmdb_structure['atomic_numbers'].dtype}, expected torch.int64 or torch.int32.")

                if "pos" not in lmdb_structure:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'pos'.")
                if not isinstance(lmdb_structure["pos"], torch.Tensor):
                    raise TypeError(f"Key {key} in the lmdb file has 'pos' of type "
                                    f"{type(lmdb_structure['pos'])}, expected torch.Tensor.")
                if not torch.is_floating_point(lmdb_structure["pos"]):
                    raise TypeError(f"Key {key} in the lmdb file has 'pos' of dtype "
                                    f"{lmdb_structure['pos'].dtype}, expected floating point dtype.")

                metadata = {"file_path": str(existing_lmdb_path), "file_key": key}
                if "ids" in lmdb_structure and "identifier" in lmdb_structure:
                    raise KeyError(f"Key {key} in the lmdb file contains both 'ids' and 'identifier'. Only one of "
                                   f"these should be present.")
                # Included for backwards compatibility.
                if "ids" in lmdb_structure:
                    metadata["identifier"] = lmdb_structure["ids"]
                if "identifier" in lmdb_structure:
                    metadata["identifier"] = lmdb_structure["identifier"]

                for prop in property_keys:
                    if prop not in lmdb_structure:
                        raise KeyError(f"Key {key} in the lmdb file does not contain '{prop}'. "
                                       f"Available keys are {list(lmdb_structure.keys())}.")

                # Test whether the structure can be created.
                structure = Structure.from_dictionary(lmdb_structure, property_keys, metadata, pos_is_fractional=False)
                if return_structures:
                    structures.append(structure)
                else:
                    structures.append(enc_key)

        return existing_lmdb_path, structures

    @staticmethod
    def _from_csv(existing_csv_path: Path, property_keys: Sequence[str], return_structures: bool,
                  cdvae_preprocessing: bool = True,
                  mattergen_preprocessing: bool = False) -> tuple[Path, Union[list[Structure], list[bytes]]]:
        """
        Check proper format of the CSV file for crystalline structures, convert to LMDB, and return the LMDB path and
        the structures or their keys within the LMDB file.

        The CSV file is expected to contain a "cif" column with the CIF representation of the structures. This will be
        used to infer the cell, atomic numbers, and positions of the structures.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the cache LMDB file in which the structures are stored (see below).
        - "file_key": The index in the LMDB file corresponding to the structure.
        - "identifier": If a "material_id" column is present in the CSV file, it will be stored here. This is usually a
                        unique identifier of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the CSV file into the structures by specifying them in the
        property_keys parameter. These keys will be read from the CSV file and stored in the property_dict of the
        structure.

        The structures from the CSV files can optionally be preprocessed to match the conventions used in CDVAE (that
        are also used in DiffCSP and FlowMM) or MatterGen.

        The structures are converted to the LMDB format for efficient access. The LMDB cache file is created in the same
        directory as the CSV file and is named based on the hash of the CSV file and the preprocessing options used.
        If the cache file already exists, it is used directly instead of creating a new one.

        In order to verify the correctness of the CSV file, this method reads all records and checks for the presence
        and correct types of the required keys. If return_structures is set to True, the created structures are
        returned. If return_structures is set to False, only the keys of the structures in the LMDB file are returned.

        :param existing_csv_path:
            Path to the CSV file.
        :type existing_csv_path: Path
        :param property_keys:
            A sequence of property keys that are read from the CSV file and stored within the structures.
        :type property_keys: Sequence[str]
        :param return_structures:
            Whether to return the structures read from the CSV file.
        :type return_structures: bool
        :param cdvae_preprocessing:
            Whether to preprocess the structures to match the conventions used in CDVAE.
            This applies the Niggli reduction of PyMatgen.
            Defaults to True.
        :type cdvae_preprocessing: bool
        :param mattergen_preprocessing:
            Whether to preprocess the structures to match the conventions used in MatterGen.
            This applies the primitive cell extraction followed by the Niggli reduction of PyMatgen.
            Defaults to False.
        :type mattergen_preprocessing: bool

        :return:
            The path of the LMDB file.
            The structures in the LMDB file if return_structures is True, otherwise the keys of the structures.
        :rtype: tuple[Path, Union[list[Structure], list[bytes]]]

        :raises ValueError:
            If both cdvae_preprocessing and mattergen_preprocessing are enabled at the same time.
        :raises KeyError:
            If any of the required keys are missing in the CSV records.
        """
        if cdvae_preprocessing and mattergen_preprocessing:
            raise ValueError("CDVAE and MatterGen preprocessing cannot both be enabled at the same time.")

        # Find or create cache file.
        # The cache file is identified by the hash of the CSV file and the preprocessing options.
        file_hash = StructureDataset._compute_file_hash(existing_csv_path, algorithm="sha256")
        cache_parts = [file_hash]
        if cdvae_preprocessing:
            cache_parts.append("cdvae")
        if mattergen_preprocessing:
            cache_parts.append("mattergen")
        if property_keys is not None:
            for prop in sorted(property_keys):
                cache_parts.append(prop)
        cache_identifier = "_".join(cache_parts)
        cache_file = existing_csv_path.with_suffix(f".csv.{cache_identifier}.lmdb")

        if cache_file.exists():
            return StructureDataset._from_lmdb(cache_file, property_keys, return_structures)

        # Check for correct columns.
        csv_columns = pd.read_csv(existing_csv_path, nrows=0).columns
        if "cif" not in csv_columns:
            raise KeyError(f"CSV file does not contain 'cif' column.")
        if property_keys is not None:
            for prop in property_keys:
                if prop not in csv_columns:
                    raise KeyError(f"CSV file does not contain '{prop}' column. "
                                   f"Available columns are {[c for c in csv_columns]}.")

        # Make sure to only read one row at a time to avoid reading the entire file into memory.
        csv_structures = pd.read_csv(existing_csv_path, chunksize=1)
        number_structures = 0
        changed_structures = 0
        structures = []
        with (lmdb.Environment(str(cache_file), subdir=False, map_size=int(1e12), lock=False) as env,
              env.begin(write=True) as txn):
            for structure_index, structure_chunk in tqdm(enumerate(csv_structures),
                                                         desc=f"Loading {existing_csv_path} data",
                                                         unit=" structures"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pymatgen_structure = PymatgenStructure.from_str(structure_chunk["cif"][structure_index], fmt="cif")

                # See https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py.
                # Configurations at https://github.com/txie-93/cdvae/tree/main/conf/data are such that niggli=True and
                # primitive=False.
                if cdvae_preprocessing:
                    cdvae_structure = pymatgen_structure.get_reduced_structure()
                    # This rotates the lattice but since we use fractional coordinates, it doesn't really matter.
                    cdvae_structure = PymatgenStructure(
                        lattice=PymatgenLattice.from_parameters(*cdvae_structure.lattice.parameters),
                        species=cdvae_structure.species,
                        coords=cdvae_structure.frac_coords,
                        coords_are_cartesian=False,
                    )
                    # Verify that the structures match.
                    if StructureMatcher(ltol=1e-4, angle_tol=1e-4, stol=1e-4, scale=False).get_rms_dist(
                            cdvae_structure, pymatgen_structure) is None:
                        changed_structures += 1
                    pymatgen_structure = cdvae_structure

                # See https://github.com/microsoft/mattergen/blob/main/mattergen/common/data/dataset.py.
                if mattergen_preprocessing:
                    # Note that this is also effectively called when one passes primitive=True to from_str.
                    mattergen_structure = pymatgen_structure.get_primitive_structure()
                    mattergen_structure = mattergen_structure.get_reduced_structure()
                    # Verify that the structures match.
                    if StructureMatcher(ltol=1e-4, angle_tol=1e-4, stol=1e-4, scale=False).get_rms_dist(
                            mattergen_structure, pymatgen_structure) is None:
                        changed_structures += 1
                    pymatgen_structure = mattergen_structure

                pos = torch.tensor(pymatgen_structure.cart_coords)
                cell = torch.tensor(pymatgen_structure.lattice.matrix)
                atomic_numbers = torch.tensor(pymatgen_structure.atomic_numbers, dtype=torch.int64)

                metadata = {"file_path": str(cache_file), "file_key": structure_index}
                if "material_id" in csv_columns:
                    metadata["identifier"] = structure_chunk["material_id"][structure_index]

                property_dict = {}
                for prop in property_keys:
                    property_dict[prop] = torch.tensor(structure_chunk[prop][structure_index])

                structure = Structure(cell=cell, atomic_numbers=atomic_numbers, pos=pos, property_dict=property_dict,
                                      metadata=metadata, pos_is_fractional=False)
                structure_key = str(structure_index).encode()
                if return_structures:
                    structures.append(structure)
                else:
                    structures.append(structure_key)

                # to_dictionary always returns Cartesian positions.
                lmdb_structure = structure.to_dictionary()
                txn.put(structure_key, pickle.dumps(lmdb_structure))

                number_structures += 1

        if changed_structures > 0:
            warnings.warn(f"{changed_structures} structures were changed during preprocessing.")

        return cache_file, structures

    @staticmethod
    def _from_parquet(existing_parquet_path: Path, property_keys: Sequence[str],
                      return_structures: bool) -> tuple[Path, Union[list[Structure], list[bytes]]]:
        """
        Check proper format of the Parquet file for crystalline structures, convert to LMDB, and return the LMDB path
        and the structures or their keys within the LMDB file.

        The Parquet file is expected to contain the following columns to represent a structure with N atoms:
        - "cell": 3x3 array-like of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "positions": Nx3 array-like of Cartesian atomic positions.
        - "atomic_numbers": array-like of N integers giving the atomic numbers of the atoms.

         The metadata of each structure will contain the following keys:
        - "file_path": The path to the cache LMDB file from in which the structures are stores (see below).
        - "file_key": The index in the LMDB file corresponding to the structure.
        - "identifier": If an "ids" or "identifier" column is present in the Parquet file, it will be stored here. This
                        is usually a unique identifier of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the Parquet file into the structures by specifying them in
        the property_keys parameter. These keys will be read from the Parquet file and stored in the property_dict of
        the structure.

        The structures are converted to the LMDB format for efficient access. The LMDB cache file is created in the same
        directory as the Parquet file and is named based on the hash of the Parquet file. If the cache file already
        exists, it is used directly instead of creating a new one.

        In order to verify the correctness of the Parquet file, this method reads all rows and checks for the presence
        and correct types of the required keys. If return_structures is set to True, the created structures are
        returned. If return_structures is set to False, only the keys of the structures in the LMDB file are returned.

        :param existing_parquet_path:
            Path to the Parquet file.
        :type existing_parquet_path: Path
        :param property_keys:
            A sequence of property keys that are read from the CSV file and stored within the structures.
        :type property_keys: Sequence[str]
        :param return_structures:
            Whether to return the structures read from the CSV file.
        :type return_structures: bool

        :return:
            The path of the LMDB file.
            The structures in the LMDB file if return_structures is True, otherwise the keys of the structures.
        :rtype: tuple[Path, Union[list[Structure], list[bytes]]]

        :raises KeyError:
            If any of the required keys are missing in the Parquet records.
        """
        file_hash = StructureDataset._compute_file_hash(existing_parquet_path, algorithm="sha256")
        cache_file = existing_parquet_path.with_suffix(f".parquet.{file_hash}.lmdb")

        if cache_file.exists():
            return StructureDataset._from_lmdb(cache_file, property_keys, return_structures)

        structures = []
        with (pq.ParquetFile(existing_parquet_path) as parquet_file,
              lmdb.Environment(str(cache_file), subdir=False, map_size=int(1e12), lock=False) as env,
              env.begin(write=True) as txn):
            # Check for correct columns.
            schemas = parquet_file.schema_arrow
            columns = [s.name for s in schemas]
            if "positions" not in columns:
                raise KeyError(f"Parquet file does not contain 'positions' column.")
            if "cell" not in columns:
                raise KeyError(f"Parquet file does not contain 'cell' column.")
            if "atomic_numbers" not in columns:
                raise KeyError(f"Parquet file does not contain 'atomic_numbers' column.")
            if property_keys is not None:
                for prop in property_keys:
                    if prop not in columns:
                        raise KeyError(f"Parquet file does not contain '{prop}' column. "
                                       f"Available columns are {columns}.")

            # Make sure to only read one row at a time to avoid reading the entire file into memory.
            number_structures = parquet_file.metadata.num_rows
            for batch_index, batch in tqdm(enumerate(parquet_file.iter_batches(batch_size=1)), total=number_structures):
                # noinspection PyUnresolvedReferences
                df = batch.to_pandas()
                assert len(df) == 1
                atomic_numbers = torch.from_numpy(np.array(df["atomic_numbers"].iloc[0], dtype=np.int64))
                assert len(atomic_numbers.shape) == 1
                pos = torch.from_numpy(np.stack(df["positions"].iloc[0]))
                assert pos.shape == (atomic_numbers.shape[0], 3)
                cell = torch.from_numpy(np.stack(df["cell"].iloc[0]))
                assert cell.shape == (3, 3)

                metadata = {"file_path": str(cache_file), "file_key": batch_index}
                if "ids" in df.columns:
                    metadata["identifier"] = df["ids"].iloc[0]
                if "identifier" in df.columns:
                    metadata["identifier"] = df["identifier"].iloc[0]
                if "material_id" in df.columns:
                    metadata["identifier"] = df["material_id"].iloc[0]

                property_dict = {}
                for prop in property_keys:
                    property_dict[prop] = torch.tensor(df[prop].iloc[0])

                structure = Structure(cell=cell, atomic_numbers=atomic_numbers, pos=pos, property_dict=property_dict,
                                      metadata=metadata, pos_is_fractional=False)
                structure_key = str(batch_index).encode()
                if return_structures:
                    structures.append(structure)
                else:
                    structures.append(structure_key)

                # to_dictionary always returns Cartesian positions.
                lmdb_structure = structure.to_dictionary()
                txn.put(structure_key, pickle.dumps(lmdb_structure))

        assert len(structures) == number_structures
        return cache_file, structures

    def to_lmdb(self, lmdb_path: Path) -> None:
        """
        Save the structures to an LMDB file.

        :param lmdb_path:
            Path to the LMDB file.
        :type lmdb_path: Path

        :raises FileExistsError:
            If the LMDB file already exists.
        """
        if lmdb_path.exists():
            raise FileExistsError(f"LMDB path {lmdb_path} already exists.")

        with (lmdb.Environment(str(lmdb_path), subdir=False, map_size=int(1e12), lock=False) as env,
              env.begin(write=True) as txn):
            for idx in tqdm(range(self._number_structures), desc=f"Saving structures to {lmdb_path}",
                            unit=" structures"):
                structure = self[idx]
                # to_dictionary always returns Cartesian positions.
                lmdb_structure = structure.to_dictionary()
                txn.put(str(idx).encode(), pickle.dumps(lmdb_structure))

    def to_xyz(self, xyz_path: Path) -> None:
        """
        Save the structures to an XYZ file.

        :param xyz_path:
            Path to the XYZ file.
        :type xyz_path: Path

        :raises FileExistsError:
            If the XYZ file already exists.
        """
        if xyz_path.exists():
            raise FileExistsError(f"XYZ path {xyz_path} already exists.")

        for idx in tqdm(range(self._number_structures), desc=f"Saving structures to {xyz_path}", unit=" structures"):
            structure = self[idx]
            atoms = structure.get_ase_atoms()
            write(str(xyz_path), atoms, format="extxyz", append=True)

    def __len__(self) -> int:
        """
        Get the number of structures within the dataset.

        This method is required to make the dataset compatible with PyTorch dataloaders.

        :return:
            The number of structures within the dataset.
        :rtype: int
        """
        return self._number_structures

    def __getitem__(self, idx: int) -> Structure:
        """
        Return the structure at the given index, either from lazy storage or from memory.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure
        """
        if self._lazy_storage:
            return self._getitem_lazy(idx)
        else:
            return self._structures[idx]

    def _getitem_lazy(self, idx: int) -> Structure:
        """
        Return the structure at the given index using lazy storage.

        The structure is read lazily from the LMDB file of this class.

        This method is only used if lazy_storage is set to True during initialization of this class. Otherwise,
        the __getitem__ method returns structures directly from the in-memory self._structures list.

        Each record (key) in the LMDB file is expected to contain a pickled dictionary with at least the following keys
        to represent a structure with N atoms:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the LMDB file from which the structure was loaded.
        - "file_key": The key in the LMDB file corresponding to the structure.
        - "identifier": If "ids" or "identifier" is present in the LMDB record, it will be stored here. This is usually
                        a unique identifier of the structure, e.g., from Materials Project.

        Additionally, the property keys will be read from the LMDB record and stored in the property_dict of the
        structure.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure

        :raises IndexError:
            If the given index is out of bounds.
        """
        if idx < 0 or idx >= self._number_structures:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self._number_structures}.")

        # Read structure from LMDB file.
        structure_idx = self._structures[idx]
        with self._env.begin(write=False) as txn:
            lmdb_structure = pickle.loads(txn.get(structure_idx))

        metadata = {"file_path": str(self._file_path), "file_key": idx}
        assert not ("ids" in lmdb_structure and "identifier" in lmdb_structure)
        if "ids" in lmdb_structure:
            metadata["identifier"] = lmdb_structure["ids"]
        if "identifier" in lmdb_structure:
            metadata["identifier"] = lmdb_structure["identifier"]

        structure = Structure.from_dictionary(lmdb_structure, self._property_keys, metadata, pos_is_fractional=False)
        if self._niggli_reduce:
            structure.niggli_reduce()
        if self._convert_to_fractional:
            structure.convert_to_fractional()
        structure.to(self._torch_precision)

        return structure

    def __getstate__(self) -> dict[str, Any]:
        """
        Return the state of the object for pickling.

        This method removes the LMDB environment from the state to avoid pickling issues. This is required for
        dataloaders with multiple workers.

        :return:
            The state of the object.
        :rtype: dict[str, Any]
        """
        state = self.__dict__.copy()
        # Remove the LMDB environment from the state to avoid pickling issues.
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the object from pickling.

        This method restores the LMDB environment from the state after unpickling. This is required for dataloaders
        with multiple workers.

        :param state:
            The state of the object.
        :type state: dict[str, Any]
        """
        self.__dict__.update(state)
        if self._lazy_storage:
            self._env = lmdb.Environment(str(self._file_path), subdir=False, readonly=True, lock=False)
        else:
            self._env = None

    def __del__(self) -> None:
        """
        Close the LMDB environment when the object is deleted.
        """
        if hasattr(self, "_env") and getattr(self, "_env") is not None:
            self._env.close()


class OverfittingDataset(StructureDataset):
    """
    Dataset that always returns the same crystalline structure from a file.

    This dataset is useful for overfitting tests. It expects the same LMDB or CSV file formats as StructureDataset.

    :param file_path:
        Path to the file containing the structures.
        This can either be an absolute path, a relative path to the current working directory, or a relative path within
        the omg package.
        Supported formats are .lmdb and .csv.
    :type file_path: str
    :param structure_index:
        Index of the structure in the LMDB file to always return.
        Defaults to 0.
    :type structure_index: int
    :param property_keys:
        An optional sequence of property keys that are read from the LMDB file and stored within the
        structures.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param floating_point_precision:
        Floating point precision to use in the structures. Can be one of "64-true", "64", "32-true", "32", "16-true",
        "16", "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        Defaults to "64-true".
    :type floating_point_precision: Union[int, str, None]
    :param parser_kwargs:
        Additional keyword arguments to pass to the specific file format parser methods.
    :type parser_kwargs: Any

    :raises KeyError:
        If the structure_index is out of bounds.
    """

    def __init__(self, file_path: str, structure_index: int = 0, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true", **parser_kwargs: Any) -> None:
        """Constructor for the OverfittingDataset class."""
        super().__init__(file_path=file_path, property_keys=property_keys,
                         floating_point_precision=floating_point_precision, **parser_kwargs)
        if not 0 <= structure_index < len(self):
            raise KeyError(f"Invalid structure index {structure_index}, "
                           f"possible values are 0 to {len(self) - 1}.")
        self._structure_index = structure_index

    def __getitem__(self, idx: int) -> Structure:
        """
        Return the structure at the given index.

        This method ignores the given index and always returns the same structure.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure
        """
        return super().__getitem__(self._structure_index)


if __name__ == '__main__':
    dataset_parquet = StructureDataset(file_path="data/Alex-MP-20_Polymorph_Split/test.parquet", lazy_storage=False,
                                       niggli_reduce=False, convert_to_fractional=True)
    dataset_lmdb = StructureDataset(file_path="data/alex_mp_20/test.lmdb", lazy_storage=False, niggli_reduce=False,
                                    convert_to_fractional=True)
    found = 0
    for parquet in tqdm(dataset_parquet):
        lmdbs = [lmdb for lmdb in dataset_lmdb if lmdb.metadata["identifier"] == parquet.metadata["identifier"]]
        if len(lmdbs) > 0:
            assert len(lmdbs) == 1
            lmdb = lmdbs[0]
            assert all(ca == la for ca, la in zip(parquet.atomic_numbers, lmdb.atomic_numbers))
            csv_structure = parquet.get_pymatgen_structure()
            lmdb_structure = lmdb.get_pymatgen_structure()
            sm = StructureMatcher(ltol=1e-3, angle_tol=1e-3, stol=1e-3, scale=False)
            res = sm.get_rms_dist(csv_structure, lmdb_structure)
            assert res is not None
            found += 1
    print("Found:", found)

    store = True
    lazy = False
    suffix = "lazy_second_frac" if lazy else "eager_second_frac"
    dataset_csv = StructureDataset(file_path="data/mp_20/test.csv", lazy_storage=lazy, cdvae_preprocessing=False,
                                   mattergen_preprocessing=False, niggli_reduce=False, convert_to_fractional=True)
    dataset_csv_cdvae = StructureDataset(file_path="data/mp_20/test.csv", lazy_storage=lazy, cdvae_preprocessing=True,
                                         mattergen_preprocessing=False, niggli_reduce=False, convert_to_fractional=True)
    dataset_csv_mattergen = StructureDataset(file_path="data/mp_20/test.csv", lazy_storage=lazy,
                                             cdvae_preprocessing=False, mattergen_preprocessing=True,
                                             niggli_reduce=False, convert_to_fractional=True)
    dataset_lmdb = StructureDataset(file_path="data/mp_20/test.lmdb", lazy_storage=lazy, niggli_reduce=False,
                                    convert_to_fractional=True)
    overfitting = OverfittingDataset(file_path="data/mp_20/test.lmdb", lazy_storage=lazy, structure_index=0,
                                     niggli_reduce=False, convert_to_fractional=True)

    if store:
        dataset_csv.to_xyz(Path(f"csv_test_{suffix}.xyz"))
        dataset_csv_mattergen.to_xyz(Path(f"mattergen_test_{suffix}.xyz"))
        dataset_csv_cdvae.to_xyz(Path(f"cdvae_test_{suffix}.xyz"))
        dataset_lmdb.to_xyz(Path(f"lmdb_test_{suffix}.xyz"))

    # noinspection PyTypeChecker
    for csv in tqdm(dataset_csv):
        lmdbs = [lmdb for lmdb in dataset_lmdb if lmdb.metadata["identifier"] == csv.metadata["identifier"]]
        assert len(lmdbs) == 1
        lmdb = lmdbs[0]
        assert all(ca == la for ca, la in zip(csv.atomic_numbers, lmdb.atomic_numbers))
        csv_structure = csv.get_pymatgen_structure()
        lmdb_structure = lmdb.get_pymatgen_structure()
        sm = StructureMatcher(ltol=1e-3, angle_tol=1e-3, stol=1e-3, scale=False)
        res = sm.get_rms_dist(csv_structure, lmdb_structure)
        assert res is not None

    # noinspection PyTypeChecker
    for csv in tqdm(dataset_csv_cdvae):
        lmdbs = [lmdb for lmdb in dataset_lmdb if lmdb.metadata["identifier"] == csv.metadata["identifier"]]
        assert len(lmdbs) == 1
        lmdb = lmdbs[0]
        assert all(ca == la for ca, la in zip(csv.atomic_numbers, lmdb.atomic_numbers))
        assert torch.allclose(csv.cell, lmdb.cell)
        csv_structure = csv.get_pymatgen_structure()
        lmdb_structure = lmdb.get_pymatgen_structure()
        sm = StructureMatcher(ltol=1e-3, angle_tol=1e-3, stol=1e-3, scale=False)
        res = sm.get_rms_dist(csv_structure, lmdb_structure)
        assert res is not None
