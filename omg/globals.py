"""Global constants for the OMG package."""
MAX_ATOM_NUM: int = 100
"""Largest atomic number in the materials dataset."""
NUM_SPECIES_WITH_GHOST: int = 119
"""Number of species when ghost atoms are used: real 1..118 + ghost (119). Used for DNG with ghosted data."""
SMALL_TIME: float = 1.0e-3
"""Lower bound for time during training and integration."""
BIG_TIME: float = 1.0 - SMALL_TIME
"""Upper bound for time during training and integration."""
