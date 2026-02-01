"""Simulation configuration.

Defines the configuration used to initialize the simulator: grid size,
random seed, nest position, and pheromone parameters. Defaults are set to
work well for typical runs.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SimConfig:
    """Simulation settings.

    Parameters control the spatial grid and simple pheromone-related
    behaviour. Adjust these values to configure the simulation environment.
    """
    grid_size: int = 256
    seed: int = 0
    nest: Tuple[int, int] = field(default=None)
    # timestep parameters
    deposit_amount: float = 1.0
    evap_rate: float = 0.01  # fraction evaporated per timestep (0..1)
    # Explorer turning kernel B: probabilities for turn magnitudes 1..4
    # B is a tuple (B1, B2, B3, B4). Each Bi >= 0 and sum(B) > 0.
    kernel_B: Tuple[float, float, float, float] = field(default=(0.25, 0.25, 0.25, 0.25))

    def __post_init__(self):
        # Initialize nest
        if self.nest is None:
            c = self.grid_size // 2
            self.nest = (c, c)

    def validate(self) -> None:
        """Sanity-check the configuration.

        Call this after constructing `SimConfig` to catch obvious mistakes
        (negative sizes, mis-typed nest coordinates, etc.). The method
        raises `ValueError` with a human-friendly message when something is
        wrong.
        """
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")

        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

        if (
            not isinstance(self.nest, tuple)
            or len(self.nest) != 2
            or not all(isinstance(v, int) for v in self.nest)
        ):
            raise ValueError("nest must be a tuple of two integers (x, y)")

        x, y = self.nest
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError("nest position must be inside the grid bounds")

        if not (0.0 <= self.evap_rate <= 1.0):
            raise ValueError("evap_rate must be between 0 and 1")

        if not (self.deposit_amount >= 0.0):
            raise ValueError("deposit_amount must be non-negative")

        if (
            not isinstance(self.kernel_B, tuple)
            or len(self.kernel_B) != 4
            or not all(isinstance(v, (int, float)) for v in self.kernel_B)
        ):
            raise ValueError("kernel_B must be a tuple of four numbers (B1..B4)")

        if sum(self.kernel_B) <= 0.0:
            raise ValueError("sum(kernel_B) must be > 0")
