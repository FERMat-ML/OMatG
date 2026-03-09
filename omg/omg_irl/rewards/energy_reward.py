import logging
from typing import Literal, Optional, Sequence
import numpy as np
import tqdm
import warnings
import torch
from omg.datamodule import Structure
from omg.utils import prefixed_stdout
from .abstracts import ComputeStage, Reward

with warnings.catch_warnings(), prefixed_stdout(prefix="[MACE] "):
    warnings.simplefilter("ignore", category=UserWarning)
    from mace.calculators import mace_mp
with prefixed_stdout(prefix="[TorchSim] "):
    from torch_sim import static
    from torch_sim.autobatching import BinningAutoBatcher
    from torch_sim.models.mace import MaceModel


class EnergyReward(Reward):
    """
    Reward function that encourages lower energies per atom based on the MACE-MPA-0 model.

    This reward first identifies invalid structures based on volume, interatomic distances, and polar sine criteria.
    If specified, these invalid structures are assigned a penalty energy instead of their computed (likely diverging)
    energies. The energy per atom of valid structures is then calculated using the MACE-MPA-0 model.

    The reward for each structure is the negative of its (possibly penalized) energy per atom, scaled by the
    provided scaling factor.

    When using a CUDA device, the MACE calculations are batched using TorchSim's BinningAutoBatcher for
    improved performance. This requires a max_memory_scaler parameter to control the batching behavior which is
    essential for managing GPU memory usage. Larger values of max_memory_scaler allow for larger batches and potentially
    better performance, but also increase the risk of out-of-memory errors. The optimal value for max_memory_scaler
    depends on the specific GPU, the size of the structures being evaluated, the floating point precision, and the
    number of structures being evaluated (since TorchSim preloads all structures as tensors before batching). The
    default value of 500000.0 is adjusted for 80GB H100 GPUs, the MP20 dataset, float64 precision, and 1024 structures.

    :param scale:
        Scaling factor for the reward.
        Must be positive.
        Defaults to 1.0.
    :type scale: float
    :param device:
        The device to run MACE on. Can be "cpu" or "cuda".
        Defaults to "cpu".
    :type device: Literal["cpu", "cuda"]
    :param default_dtype:
        The default dtype to use for MACE calculations. Can be "float32" or "float64".
        Defaults to "float64".
    :type default_dtype: Literal["float32", "float64"]
    :param enable_cueq:
        Whether to enable the CuEq in MACE.
        Defaults to False.
    :type enable_cueq: bool
    :param max_memory_scaler:
        The max_memory_scaler parameter for TorchSim's BinningAutoBatcher when using CUDA.
        Defaults to 500000.0, which is adjusted for 80GB A100 GPUs and the MP20 dataset.
    :type max_memory_scaler: float
    :param invalid_penalty:
        Penalty energy in eV to assign to invalid structures.
        If None, no penalty is applied and energies for invalid structures are computed normally.
        Defaults to None.
    :type invalid_penalty: Optional[float]
    :param volume_check_cutoff:
        Minimum volume for which to compute the energy normally.
        If the generated structure has a volume below this cutoff, the energy is potentially penalized.
        Must be non-negative.
        Defaults to 0.1.
    :type volume_check_cutoff: float
    :param structure_check_cutoff:
        If the minimum interatomic distance in the generated structure is below this cutoff, the energy is potentially
        penalized.
        Must be non-negative.
        Defaults to 0.5.
    :type structure_check_cutoff: float
    :param polar_sine_cutoff:
        If the polar sine (volume / product of lattice lengths) of the generated structure is below this cutoff,
        the energy is potentially penalized.
        Must be non-negative.
        Defaults to 1.0e-3.
    :type polar_sine_cutoff: float

    :raises ValueError:
        If scale is not positive.
        If volume_check_cutoff is negative.
        If structure_check_cutoff is negative.
        If polar_sine_cutoff is negative.
    """

    def __init__(self, scale: float = 1.0, device: Literal["cpu", "cuda"] = "cpu",
                 default_dtype: Literal["float32", "float64"] = "float64", enable_cueq: bool = False,
                 max_memory_scaler: float = 500000.0, invalid_penalty: Optional[float] = None,
                 volume_check_cutoff: float = 0.1, structure_check_cutoff: float = 0.5,
                 polar_sine_cutoff: float = 1.0e-3) -> None:
        """Constructor for EnergyReward."""
        super().__init__()
        if not scale > 0.0:
            raise ValueError("Scale must be positive.")
        if not volume_check_cutoff >= 0.0:
            raise ValueError("Volume check cutoff must be non-negative.")
        if not structure_check_cutoff >= 0.0:
            raise ValueError("Structure check cutoff must be non-negative.")
        if not polar_sine_cutoff >= 0.0:
            raise ValueError("Polar sine cutoff must be non-negative.")
        self._scale = scale
        self._device = device
        self._default_dtype = default_dtype
        # Catch warnings from MACE and prefix stdout.
        with prefixed_stdout("[MACE] "), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if self._device == "cpu":
                logging.disable(logging.WARNING)  # MACE prints a warning when using CPU, disable it.
                # Calling mace_mp with return_raw_model=False changes the default dtype of torch.
                self._mace_model = mace_mp(model="medium-mpa-0", device=device, default_dtype=default_dtype,
                                           enable_cueq=enable_cueq)
                logging.disable(logging.NOTSET)  # Undo the disabling of logging.
                torch.set_default_dtype(torch.float32)  # Undo what happened in mace_mp construction.
                self._batcher = None
                if max_memory_scaler != 500000.0:
                    warnings.warn("max_memory_scaler is only used when device is 'cuda', the specified value will be "
                                  "ignored.")
            else:
                # TorchSim's batching does not work on CPU.
                assert self._device == "cuda"
                mace = mace_mp(model="medium-mpa-0", device=device, default_dtype=default_dtype,
                               enable_cueq=enable_cueq, return_raw_model=True)
                # noinspection PyTypeChecker
                self._mace_model = MaceModel(model=mace, device=device, compute_forces=False, compute_stress=False,
                                             dtype=torch.float64 if default_dtype == "float64" else torch.float32,
                                             enable_cueq=enable_cueq)
                self._batcher = BinningAutoBatcher(model=self._mace_model, max_memory_scaler=max_memory_scaler,
                                                   memory_scales_with="n_atoms_x_density")
        self._invalid_penalty = invalid_penalty
        self._volume_check_cutoff = volume_check_cutoff
        self._structure_check_cutoff = structure_check_cutoff
        self._polar_sine_cutoff = polar_sine_cutoff

    def _is_valid(self, structure: Structure) -> bool:
        """Check if the structure is valid based on volume, interatomic distances, and polar sine criteria."""
        py_structure = structure.get_pymatgen_structure()
        volume_valid = (py_structure.volume >= self._volume_check_cutoff)
        dist_mat = py_structure.distance_matrix
        dist_mat = dist_mat + np.diag(
            np.ones(dist_mat.shape[0]) * (self._structure_check_cutoff + 10.0)
        )
        min_dist = dist_mat.min()
        structure_valid = (min_dist >= self._structure_check_cutoff)
        polar_sine = py_structure.lattice.volume / np.prod(py_structure.lattice.lengths)
        polar_sine_valid = (polar_sine >= self._polar_sine_cutoff)
        return volume_valid and structure_valid and polar_sine_valid

    def compute(self, structures: Sequence[Structure], stage: ComputeStage,
                enable_progress_bar: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute energy rewards for a batch of structures.

        The stage parameter is included for compatibility but is not used in this reward function.

        :param structures:
            Sequence of Structure objects representing generated structures.
        :type structures: Sequence[Structure]
        :param stage:
            Stage of the reward computation. Not used in this reward function.
        :type stage: ComputeStage
        :param enable_progress_bar:
            Whether to enable the progress bar for this computation.
        :type enable_progress_bar: bool

        :return:
            (List of rewards per structure, info dictionary).
        :rtype: tuple[np.ndarray, dict[str, np.ndarray]]
        """
        energies = np.empty(len(structures), dtype=float)
        invalid_flags = np.zeros(len(structures), dtype=float)

        if self._device == "cpu":
            with torch.set_grad_enabled(True):  # Mace needs gradients.
                # Mace calculator needs appropriate default dtype.
                torch.set_default_dtype(torch.float64 if self._default_dtype == "float64" else torch.float32)
                for idx, structure in enumerate(tqdm.tqdm(structures, desc="Computing energy rewards with MACE",
                                                          disable=not enable_progress_bar, unit="structures",
                                                          position=1)):
                    if self._invalid_penalty is not None and not self._is_valid(structure):
                        energies[idx] = self._invalid_penalty
                        invalid_flags[idx] = 1.0
                        continue
                    energies[idx] = (
                        self._mace_model.get_potential_energy(structure.get_ase_atoms())
                        / len(structure.atomic_numbers))
                torch.set_default_dtype(torch.float32)  # Undo the change to default dtype.
        else:
            assert self._device == "cuda"
            if self._invalid_penalty is not None:
                valid_mask = np.array([self._is_valid(structure) for structure in structures])
            else:
                valid_mask = np.ones(len(structures), dtype=bool)

            energies[~valid_mask] = self._invalid_penalty
            invalid_flags[~valid_mask] = 1.0

            valid_atoms = [structures[i].get_ase_atoms() for i in range(len(structures)) if valid_mask[i]]
            res = static(system=valid_atoms, model=self._mace_model, autobatcher=self._batcher,
                         pbar={"desc": "Computing valid energy rewards with MACE",
                               "postfix": f"{len(structures) - len(valid_atoms)} invalid structures",
                               "unit": "structures", "position": 1, "leave": False} if enable_progress_bar else False)
            assert len(res) == len(valid_atoms)
            energies[valid_mask] = [float(r["potential_energy"][0]) / len(atoms)
                                    for r, atoms in zip(res, valid_atoms)]

        rewards = -self._scale * energies  # Minimize energy.
        info_dict = {"energy_per_atom": energies, "energy_invalid": invalid_flags}
        return rewards, info_dict
