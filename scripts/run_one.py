#!/usr/bin/env python3
"""Runner for the ant simulator.

Creates a `SimConfig` and `SimState`, spawns one ant at the nest, and
advances the model for several timesteps. The script prints the grid
shape and a step-by-step log of the ant's positions and directions.

Usage::

    python scripts/run_one.py
"""
import os
import sys

# Make `src` importable when running from repo root (scripts/ is sibling of src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from antsim.config import SimConfig
from antsim.state import SimState, DX, DY


def main():
    cfg = SimConfig()
    cfg.validate()
    state = SimState(cfg)
    print("Pheromone grid:", state.grid.shape, state.grid.dtype)
    print("Ant count:", state.ant_count)
    print("Nest position:", cfg.nest)
    print("Direction encoding (dx, dy):")
    print("dx =", DX.tolist())
    print("dy =", DY.tolist())

    # Spawn
    idx = state.spawn_ant(cfg.nest[0], cfg.nest[1], direction=0, mode=0)
    print("Spawned ant index:", idx)
    print("Ant count after spawn:", state.ant_count)
    print("Ant arrays (x,y,dir,mode):")
    print(state.x, state.y, state.direction, state.mode)

    # Run a few timesteps with spawning disabled to test step loop
    steps = 5
    print(f"Running {steps} timesteps...")
    for t in range(steps):
        # Step
        state.step(spawn=0)

        print(f"t={t+1}: ant_count={state.ant_count}, total_pheromone={state.grid.sum():.3f}")
        print("  positions:", state.x.tolist(), state.y.tolist(), "dirs:", state.direction.tolist())


if __name__ == "__main__":
    main()
