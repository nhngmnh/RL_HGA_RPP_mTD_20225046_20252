from __future__ import annotations

from data import HGAParams


def get_hga_params() -> HGAParams:
    """Params for the Hybrid Genetic Algorithm (HGA)."""
    return HGAParams(
        # Population
        PL=100,
        PH=200,
        # Generations
        G=100,
        Gm=10,
        # Initializer
        pt=0.1,
        # Mutation
        pm=0.1,
        pm_plus=0.3,
        # Reproducibility
        seed=42,
    )


def get_ga_params() -> HGAParams:
    """Params for the Genetic Algorithm (GA).

    Note: GA often benefits from more generations than HGA.
    """
    return HGAParams(
        # Keep the same defaults as HGA unless you want to tune further.
        PL=100,
        PH=200,
        G=1000,
        Gm=10,
        pt=0.1,
        pm=0.1,
        pm_plus=0.3,
        seed=42,
    )
