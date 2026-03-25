# Energy above convex hull computation.
#
# Adapted from LeMat-GenBench (https://github.com/LeMaterial/lemat-genbench)
# Original source: src/lemat_genbench/preprocess/reference_energies.py
# Licensed under the Apache License, Version 2.0
# Copyright LeMaterial contributors
#
# Modifications:
# - Extracted only the functions needed for energy above hull computation.
# - Removed local file path fallbacks relative to the lemat_genbench package directory.
# - Simplified HuggingFace Hub download to use the default cache directory.

import logging
from collections import Counter
from functools import lru_cache

import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from scipy import sparse

_logger = logging.getLogger(__name__)


def _one_hot_encode_composition(elements) -> np.ndarray:
    """One-hot encode a list of elements into a vector of length 119 (indexed by atomic number)."""
    one_hot = np.zeros(119)  # 0 unused, 1-118 for elements.
    for element in elements:
        # Handle charged species by extracting the base element.
        if isinstance(element, str):
            if "+" in element or "-" in element:
                base_element = element.rstrip("+-0123456789")
            else:
                base_element = element
        elif hasattr(element, "element"):
            base_element = element.element
        else:
            base_element = element
        element_obj = Element(base_element)
        one_hot[int(element_obj.number)] = 1
    return one_hot


@lru_cache(maxsize=None)
def _retrieve_df(hull_type: str = "mace_omat", threshold: float = 0.001):
    """Retrieve the reference dataset for hull computations from HuggingFace Hub.

    Tries loading via the ``datasets`` library first (``LeMaterial/LeMat-Bulk-MLIP-Hull``),
    and falls back to downloading a parquet file via ``huggingface_hub``.

    The result is cached so subsequent calls with the same arguments are free.
    """
    import pandas as pd

    # Primary: load via datasets library.
    try:
        from datasets import load_dataset
        dataset_dict = load_dataset("LeMaterial/LeMat-Bulk-MLIP-Hull")
        if hull_type in dataset_dict:
            dataset = dataset_dict[hull_type].to_pandas()
            if "species_at_sites" in dataset.columns:
                dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
            return dataset
    except Exception:
        pass

    # Fallback: download parquet file directly.
    from huggingface_hub import hf_hub_download

    threshold_str = f"{threshold:.3f}".replace(".", "_")
    file_path = hf_hub_download(
        repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
        filename=f"threshold_{threshold_str}/{hull_type}_above_hull_dataset.parquet",
        repo_type="dataset",
    )
    dataset = pd.read_parquet(file_path)
    if "elements" in dataset.columns:
        dataset["elements"] = dataset["elements"].apply(
            lambda x: x.tolist() if hasattr(x, "tolist") else x
        )
    if "species_at_sites" in dataset.columns:
        dataset["species_at_sites"] = dataset["species_at_sites"].apply(
            lambda x: x.tolist() if hasattr(x, "tolist") else x
        )
    return dataset


@lru_cache(maxsize=None)
def _retrieve_matrix(hull_type: str = "mace_omat", threshold: float = 0.001) -> np.ndarray:
    """Retrieve the composition matrix for hull computations from HuggingFace Hub.

    The result is cached so subsequent calls with the same arguments are free.
    """
    from huggingface_hub import hf_hub_download

    threshold_str = f"{threshold:.3f}".replace(".", "_")
    file_path = hf_hub_download(
        repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
        filename=f"threshold_{threshold_str}/{hull_type}_above_hull_composition_matrix.npz",
        repo_type="dataset",
    )
    return sparse.load_npz(file_path).toarray()


def _filter_df(df, matrix: np.ndarray, composition: Composition):
    """Filter the reference dataset to entries whose elements are a subset of the given composition."""
    structure_vector = _one_hot_encode_composition(composition.elements).reshape(-1, 1)
    forbidden_elements = 1 - structure_vector
    return df.loc[(matrix @ forbidden_elements) == 0]


def _neutralize_composition(composition: Composition) -> Composition:
    """Convert a composition with potentially charged species to a neutral composition."""
    neutral = {}
    for element, count in composition.as_dict().items():
        if isinstance(element, str):
            if "+" in element or "-" in element:
                base_element = element.rstrip("+-0123456789")
            else:
                base_element = element
        elif hasattr(element, "element"):
            base_element = element.element
        else:
            base_element = element
        neutral[base_element] = neutral.get(base_element, 0) + count
    return Composition(neutral)


def get_energy_above_hull(total_energy: float, composition: Composition,
                          hull_type: str = "mace_omat", threshold: float = 0.001) -> float:
    """Calculate energy above the convex hull from total energy and composition.

    Parameters
    ----------
    total_energy : float
        Total energy in eV.
    composition : Composition
        Pymatgen Composition object (may contain charged species).
    hull_type : str
        Reference hull type. One of 'mace_mp', 'mace_omat', 'dft', 'orb', 'uma'.
        Defaults to 'mace_omat'.
    threshold : float
        Energy above hull threshold in eV/atom for filtering the reference dataset.
        Defaults to 0.001.

    Returns
    -------
    float
        Energy above hull in eV/atom (intensive, following Materials Project conventions).

    Raises
    ------
    ValueError
        If the hull computation fails (e.g., no reference structures for the given elements).
    """
    try:
        df = _retrieve_df(hull_type, threshold)
        matrix = _retrieve_matrix(hull_type, threshold)
        intersection = _filter_df(df, matrix, composition)

        pd_entries = [
            PDEntry(Composition(Counter(row["species_at_sites"])), row["energy"])
            for _, row in intersection.iterrows()
        ]
        if not pd_entries:
            raise ValueError(
                f"No entries found in dataset containing any of the elements in: {composition.elements}"
            )

        phase_diagram = PhaseDiagram(pd_entries)
        neutral_composition = _neutralize_composition(composition)
        entry = PDEntry(neutral_composition, total_energy)
        e_above_hull = phase_diagram.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]
        return e_above_hull

    except Exception as e:
        raise ValueError(
            f"Failed to compute energy above hull for {composition.formula}: {e}"
        ) from e
