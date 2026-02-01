"""Simulation state: pheromone grid and compact ant state arrays.

This module provides a memory- and performance-friendly storage layout
for ant positions/directions/mode alongside a 2D pheromone concentration
grid implemented with NumPy.
"""
from typing import Tuple
import numpy as np

from .config import SimConfig

# Direction encoding: 0 = east, then counterclockwise
# dx and dy can be used to update positions: new_x = x + dx[dir]
"""Simulation state and helper routines.

This module implements the core state storage for the simulator. Ants are
represented by compact NumPy arrays and the pheromone field is a 2D NumPy
array. The layout is chosen for efficient, vectorized updates.

Note
----
The pheromone grid is `grid[y, x]` (dtype `float32`). Ant arrays
(`x`, `y`, `direction`, `mode`) are parallel 1D arrays where each index
represents one ant.
"""
from typing import Tuple
import numpy as np

from .config import SimConfig

# Direction encoding: 0 = east, then counterclockwise. Use these to move ants.
DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int8)
DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int8)


class SimState:
    """Holds the pheromone grid and compact ant arrays.

    The class provides a small API for adding/removing ants and advancing
    the simulation with a single `step()` call. The implementation favors
    clarity and performance using NumPy operations.
    """

    def __init__(self, config: SimConfig):
        config.validate()
        self.config = config

        # pheromone concentration grid C[y, x]
        self.grid = np.zeros((config.grid_size, config.grid_size), dtype=np.float32)

        # Ant data stored as compact 1D arrays. Each index corresponds to one ant.
        self.x = np.empty(0, dtype=np.int32)
        self.y = np.empty(0, dtype=np.int32)
        self.direction = np.empty(0, dtype=np.uint8)
        self.mode = np.empty(0, dtype=np.uint8)

        # RNG seed
        np.random.seed(config.seed)

        # Kernel magnitudes (1..4)
        self.kernel_B_mags = np.array([1, 2, 3, 4], dtype=np.int8)

    def spawn_ant(self, x: int, y: int, direction: int, mode: int = 0) -> int:
        """Add a single ant and return its index.

        Parameters are validated to help users spot errors quickly.
        """
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            raise ValueError("ant position out of bounds")
        if not (0 <= direction <= 7):
            raise ValueError("direction must be in 0..7")
        if mode not in (0, 1):
            raise ValueError("mode must be 0 (explorer) or 1 (follower)")

        # Append ant
        self.x = np.append(self.x, np.int32(x))
        self.y = np.append(self.y, np.int32(y))
        self.direction = np.append(self.direction, np.uint8(direction))
        self.mode = np.append(self.mode, np.uint8(mode))

        return self.ant_count - 1

    def remove_ants(self, mask) -> None:
        """Drop ants where `mask` is True.

        The `mask` must be a boolean array with the same length as the number
        of ants.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.ant_count,):
            raise ValueError("mask must have shape (ant_count,)")

        keep = ~mask
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.direction = self.direction[keep]
        self.mode = self.mode[keep]

    @property
    def ant_count(self) -> int:
        """Return the number of ants currently in the simulation."""
        return int(self.x.size)

    def direction_encoding(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return copies of the `DX, DY` arrays if you need them externally."""
        return DX.copy(), DY.copy()

    def step(self, spawn: int = 0) -> None:
        """Advance the simulation by one timestep.

        The step order follows the paper: deposit at the ant's current cell,
        then move. The sequence is:
          1. Optionally spawn `spawn` new ants at the nest.
          2. For explorer ants, sample a new direction using kernel B.
             We sample a magnitude in 1..4 and a random sign (±), then
             update the direction modulo 8.
          3. Deposit pheromone at the ant's current location.
          4. Move ants by one cell according to their (possibly updated)
             directions using `DX` and `DY`.
          5. Remove ants that left the grid.
          6. Evaporate the pheromone grid multiplicatively.

        This method keeps operations vectorized where possible for speed.
        """
        # Spawn
        if spawn > 0:
            nx, ny = self.config.nest
            dirs = np.random.randint(0, 8, size=spawn, dtype=np.uint8)
            xs = np.full(spawn, nx, dtype=np.int32)
            ys = np.full(spawn, ny, dtype=np.int32)
            modes = np.zeros(spawn, dtype=np.uint8)

            self.x = np.concatenate((self.x, xs))
            self.y = np.concatenate((self.y, ys))
            self.direction = np.concatenate((self.direction, dirs))
            self.mode = np.concatenate((self.mode, modes))

        if self.ant_count == 0:
            # No ants to update — just evaporate the grid and return.
            self._evaporate()
            return

        # Turn (kernel B)
        explorers = (self.mode == 0)
        if np.any(explorers):
            e_idx = np.nonzero(explorers)[0]
            n_e = e_idx.size
            weights = np.asarray(self.config.kernel_B, dtype=np.float64)
            weights = weights / float(weights.sum())

            # Sample magnitudes and signs, then apply as relative turns.
            mags = np.random.choice(self.kernel_B_mags, size=n_e, p=weights)
            signs = np.random.choice(np.array([-1, 1], dtype=np.int8), size=n_e)
            rel_turns = (mags * signs).astype(np.int8)

            new_dirs = (self.direction[e_idx].astype(np.int8) + rel_turns) % 8
            self.direction[e_idx] = new_dirs.astype(np.uint8)

        # Deposit
        yy = self.y.astype(np.intp)
        xx = self.x.astype(np.intp)
        np.add.at(self.grid, (yy, xx), float(self.config.deposit_amount))

        # Move
        self.x = (self.x + DX[self.direction].astype(np.int32)).astype(np.int32)
        self.y = (self.y + DY[self.direction].astype(np.int32)).astype(np.int32)

        # Remove off-grid
        g = self.config.grid_size
        off = (self.x < 0) | (self.x >= g) | (self.y < 0) | (self.y >= g)
        if np.any(off):
            self.remove_ants(off)

        # Evaporate
        self._evaporate()

    def _evaporate(self) -> None:
        """Evaporate pheromone grid by `(1 - evap_rate)`."""
        rate = float(self.config.evap_rate)
        if rate <= 0.0:
            return
        if rate >= 1.0:
            # Everything evaporates immediately.
            self.grid.fill(0.0)
            return
        self.grid *= (1.0 - rate)
