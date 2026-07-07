"""Shared constants for validator emission handling."""

# Emission burning mechanism
BURN_EMISSIONS = True              # Enable emission burning to UID 0
BURN_FRACTION = 0.00               # Fraction of emissions to burn
KEEP_FRACTION = 1.0 - BURN_FRACTION  # Fraction of emissions to distribute
UID_ZERO = 0                       # UID representing the burn address

# Reward distribution mechanism
WINNER_TAKE_ALL = True             # Winner-take-all emission distribution
