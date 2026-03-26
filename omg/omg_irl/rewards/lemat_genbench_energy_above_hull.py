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
from collections import Counter
from functools import lru_cache
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from scipy import sparse


def _one_hot_encode_composition(elements) -> np.ndarray:
    one_hot = np.zeros(
        119
    )  # 119 to use direct Z indexing (0 unused, 1-118 for elements)
    for element in elements:
        try:
            # Handle charged species by extracting the base element
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Validate element and get atomic number
            element_obj = Element(base_element)
            one_hot[int(element_obj.number)] = 1
        except ValueError as e:
            print(f"Warning: Invalid element '{element}': {e}")
            continue
    return one_hot


@lru_cache(maxsize=None)
def _retrieve_df(hull_type: str = "mace_mp", threshold: float = 0.001):
    """Retrieve the reference dataset for hull computations from HuggingFace Hub.

    Tries loading via the ``datasets`` library first (``LeMaterial/LeMat-Bulk-MLIP-Hull``),
    and falls back to downloading a parquet file via ``huggingface_hub``.

    The result is cached so subsequent calls with the same arguments are free.
    """
    file_path = hf_hub_download(
        repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
        filename=f"data/{hull_type}-00000-of-00001.parquet",
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
def _retrieve_matrix(hull_type: str = "mace_mp", threshold: float = 0.001) -> np.ndarray:
    """Retrieve the composition matrix for hull computations from HuggingFace Hub.

    The result is cached so subsequent calls with the same arguments are free.
    """
    threshold_str = f"{threshold:.3f}".replace(".", "_")
    file_path = hf_hub_download(
        repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
        filename=f"threshold_{threshold_str}/{hull_type}_above_hull_composition_matrix.npz",
        repo_type="dataset",
    )
    composition_array = sparse.load_npz(file_path).toarray()
    return composition_array


def _filter_df(df, matrix: np.ndarray, composition: Composition):
    """Filter the reference dataset to entries whose elements are a subset of the given composition."""
    try:
        structure_vector = _one_hot_encode_composition(composition.elements).reshape(
            -1, 1
        )
        forbidden_elements = 1 - structure_vector
        intersection_elements = df.loc[(matrix @ forbidden_elements) == 0]

        if intersection_elements.empty:
            print(
                f"Warning: No reference structures found for composition {composition.formula}"
            )

        return intersection_elements
    except Exception as e:
        raise ValueError(f"Failed to filter reference dataset: {e}") from e


def get_energy_above_hull(total_energy: float, composition: Composition,
                          hull_type: str = "mace_mp", threshold: float = 0.001) -> float:
    """Calculate energy above the convex hull from total energy and composition.

    Parameters
    ----------
    total_energy : float
        Total energy in eV.
    composition : Composition
        Pymatgen Composition object (may contain charged species).
    hull_type : str
        Reference hull type. One of 'mace_mp', 'mace_omat', 'dft', 'orb', 'uma'.
        Defaults to 'mace_mp'.
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
        intersection_elements = _filter_df(
            _retrieve_df(hull_type, threshold),
            _retrieve_matrix(hull_type, threshold),
            composition,
        )

        # Create PDEntries from the filtered DataFrame
        # using species_at_sites for composition because no ambiguity there
        pd_entries = [
            PDEntry(Composition(Counter(row["species_at_sites"])), row["energy"])
            for _, row in intersection_elements.iterrows()
        ]

        if not pd_entries:
            raise ValueError(
                f"No entries found in dataset containing any of the elements in: {composition.elements}"
            )

        # Construct phase diagram
        pd = PhaseDiagram(pd_entries)

        # Convert charged species to neutral composition for PDEntry creation
        # This follows the same logic as get_formation_energy_from_composition_energy
        neutral_composition_dict = {}
        for element, count in composition.as_dict().items():
            # Handle charged species by extracting the base element
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Use neutral element for PDEntry creation
            if base_element not in neutral_composition_dict:
                neutral_composition_dict[base_element] = 0
            neutral_composition_dict[base_element] += count

        neutral_composition = Composition(neutral_composition_dict)

        entry = PDEntry(neutral_composition, total_energy)
        e_above_hull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]

        return e_above_hull

    except Exception as e:
        raise ValueError(
            f"Failed to compute energy above hull for {composition.formula}: {str(e)}"
        ) from e
