# Summary of Changes in This Fork

## Overview

This fork adds comprehensive support for **ghost atoms** (non-physical placeholder atoms) throughout the OMatG codebase. Ghost atoms are preserved during training and generation but are automatically filtered out before physical validation, analysis, and metrics computation.

## Main Feature: Ghost Atom Support

### Ghost Atom Representation

Ghost atoms use a multi-format representation system:

- **Storage (LMDB)**: Atomic number `-1`
- **Model Processing**: Label `119` (one more than the highest real atomic number 118)
- **ASE Export**: `max_real_Z + 1` (dynamic based on structure)
- **Chemical Symbol**: `"Gh"` added to symbol mappings

### Design Philosophy
 
Ghost atoms are:
- ✅ Preserved during training and generation
- ✅ Automatically filtered before:
  - Physical validation checks
  - Bond/coordination calculations
  - Metrics computation
  - Visualization

## Detailed Changes by Component

### 1. Data Loading (`omg/datamodule/omg_data.py`)

**Changes:**
- Modified `OMGData.__init__()` to convert ghost atoms from `-1` to `119` when creating from Structure objects
- Ensures ghost atoms are distinct from mask tokens (0) in the model

**Key Code:**
```python
# Handle ghost atoms: convert -1 to fixed global label (119) to keep distinct from mask (0)
species = structure.atomic_numbers.clone()
ghost_mask = species < 0
if ghost_mask.any():
    GLOBAL_GHOST_LABEL = 119  # Fixed label for all ghost atoms
    species[ghost_mask] = GLOBAL_GHOST_LABEL
self.species = species
```

### 2. Model Encoders

#### `omg/model/encoders/cspnet_full.py`
**Changes:**
- Increased embedding size from `max_atoms` to `max_atoms + 20` to accommodate species label 119

**Key Code:**
```python
# Embedding size increased to accommodate ghost atoms (label 119)
# With species_shift=1, species 119 maps to index 118, so we need at least 119 entries
# Using max_atoms + 20 = 120 to safely handle species up to 119
self.node_embedding = nn.Embedding(max_atoms + 20, hidden_dim)  # Supports ghost atoms up to species 119
```

#### `omg/model/encoders/diffcsp_copies.py`
**Changes:**
- Same embedding size increase (`max_atoms + 20`) for ghost atom support in `CSPNet.__init__()`

**Key Code:**
```python
# Embedding size increased to accommodate ghost atoms (label 119)
# With species_shift=1, species 119 maps to index 118, so we need at least 119 entries
# Using max_atoms + 20 = 120 to safely handle species up to 119
if self.smooth:
    self.node_embedding = nn.Linear(max_atoms + 20, hidden_dim)
else:
    self.node_embedding = nn.Embedding(max_atoms + 20, hidden_dim)
```

### 3. Utilities (`omg/utils.py`)

**Changes:**
- Enhanced `xyz_saver()` function to detect and handle ghost atoms in XYZ file export

**Key Code:**
```python
MAX_Z = 118  # highest element ASE knows about
ghost_mask = species_slice <= 0
ghost_mask |= species_slice > MAX_Z
safe_species[ghost_mask] = 0  # map ghosts to dummy element 'X'

if ghost_mask.any():
    atom.new_array("is_ghost", ghost_mask.astype(np.int8))
    ghost_numbers = np.full_like(safe_species, -1)
    ghost_numbers[ghost_mask] = species_slice[ghost_mask]
    atom.new_array("ghost_atomic_number", ghost_numbers)
```

### 4. Training & Metrics (`omg/omg_trainer.py`)

**Changes:**
- Modified `_load_dataset_atoms()` to convert ghost atoms from `-1` to `max_real_Z + 1` for ASE compatibility and set `is_ghost` array
- Modified `_plot_to_pdf()` to filter ghost atoms before visualization and metrics computation
- Added handling for mismatched atom counts after ghost filtering

**Key Code:**
```python
# Handle ghost atoms: species=-1 should be converted to max_real_Z+1 for ASE compatibility
is_ghost = species_np == -1
if is_ghost.any():
    real_numbers = species_np[~is_ghost]
    max_real = int(real_numbers.max()) if real_numbers.size else 0
    ghost_label = max_real + 1
    numbers_for_ase[is_ghost] = ghost_label
    atoms.set_array("is_ghost", is_ghost)

# Filter ghost atoms before visualization
def filter_ghost_atoms(atoms_list: Sequence[Atoms]) -> list[Atoms]:
    """Filter ghost atoms from a list of Atoms objects."""
    filtered = []
    for atoms in atoms_list:
        if "is_ghost" in atoms.arrays:
            is_ghost = atoms.arrays["is_ghost"].astype(bool)
            atoms = atoms[~is_ghost]
        else:
            valid_mask = (atoms.numbers > 0) & (atoms.numbers <= MAX_ATOM_NUM)
            atoms = atoms[valid_mask]
        filtered.append(atoms)
    return filtered
```

### 5. Analysis & Validation

#### `omg/analysis/valid_atoms.py`
**Changes:**
- Added ghost atom filtering at the start of `__init__()` before all validation checks
- Marks structures as invalid if all atoms are ghosts

**Key Code:**
```python
# Filter out ghost atoms before validation
# Ghost atoms are non-physical and should not be included in metrics
if "is_ghost" in atoms.arrays:
    is_ghost = atoms.arrays["is_ghost"].astype(bool)
    atoms = atoms[~is_ghost]
else:
    # Fallback: filter by atomic number (ghost atoms have number <= 0 or > MAX_ATOM_NUM)
    valid_mask = (atoms.numbers > 0) & (atoms.numbers <= MAX_ATOM_NUM)
    atoms = atoms[valid_mask]

# If all atoms were ghost atoms, mark as invalid
if len(atoms) == 0:
    # Set all validation flags to False
```

#### `omg/analysis/analysis.py`
**Changes:**
- Added ghost atom filtering in `get_bonds()` before bond calculations
- Added ghost atom filtering in `get_coordination_numbers()` before coordination calculations

**Key Code:**
```python
# Filter out ghost atoms before computing bonds
# Ghost atoms are non-physical and should not be included in coordination calculations
if "is_ghost" in atoms.arrays:
    is_ghost = atoms.arrays["is_ghost"].astype(bool)
    atoms = atoms[~is_ghost]
else:
    # Fallback: filter by atomic number (ghost atoms have number <= 0 or > MAX_ATOM_NUM)
    valid_mask = (atoms.numbers > 0) & (atoms.numbers <= MAX_ATOM_NUM)
    atoms = atoms[valid_mask]

# If all atoms were ghost atoms, return empty results
if len(atoms) == 0:
    return []  # or empty list/dict as appropriate
```

## Implementation Details

### Ghost Atom Detection

Ghost atoms are identified by:
1. **Primary method**: `is_ghost` array in ASE Atoms objects (set during conversion)
2. **Fallback method**: Atomic number check (`number <= 0` or `number > MAX_ATOM_NUM`)

### Conversion Pipeline

1. **LMDB → Model**: `-1` → `119` (in `omg_data.py`)
2. **Model → ASE**: `119` → `max_real_Z + 1` (in `omg_trainer.py`)
3. **ASE Export**: Ghost atoms marked with `is_ghost` array and mapped to 'X' (0) for compatibility

### Filtering Strategy

Ghost atoms are filtered using a consistent pattern:
```python
if "is_ghost" in atoms.arrays:
    is_ghost = atoms.arrays["is_ghost"].astype(bool)
    atoms = atoms[~is_ghost]
else:
    # Fallback: filter by atomic number
    valid_mask = (atoms.numbers > 0) & (atoms.numbers <= MAX_ATOM_NUM)
    atoms = atoms[valid_mask]
```

## Benefits

1. **Training Flexibility**: Models can be trained on structures with placeholder atoms
2. **Automatic Filtering**: No manual intervention needed - ghost atoms are automatically excluded from physical analysis
3. **Backward Compatibility**: Existing code paths work correctly with ghost atoms present
4. **Consistent Handling**: Unified approach across all analysis and validation functions

## Files Modified

- `omg/datamodule/omg_data.py` - Ghost atom conversion in data loading
- `omg/utils.py` - Ghost atom handling in XYZ I/O
- `omg/omg_trainer.py` - Ghost atom filtering in training/metrics
- `omg/analysis/valid_atoms.py` - Ghost atom filtering in validation
- `omg/analysis/analysis.py` - Ghost atom filtering in bond/coordination calculations
- `omg/model/encoders/cspnet_full.py` - Embedding size for ghost atoms
- `omg/model/encoders/diffcsp_copies.py` - Embedding size for ghost atoms

## Notes

- All formatting-only changes have been excluded from this summary
- Only substantive code changes related to ghost atom functionality are documented
- The implementation maintains compatibility with existing non-ghosted datasets

