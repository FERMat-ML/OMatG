"""Global constants for the OMG package."""
MAX_ATOM_NUM: int = 100
"""Largest atomic number in the materials dataset."""
SMALL_TIME: float = 1.0e-3
"""Lower bound for time during training and integration."""
BIG_TIME: float = 1.0 - SMALL_TIME
"""Upper bound for time during training and integration."""
