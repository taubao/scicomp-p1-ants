"""Unit tests"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.antsim as antsim


# Make stochastic functions deterministic for tests.
antsim.rng = np.random.default_rng(0)


def test_phi_linear_bounds():
    assert antsim.phi_linear(0, phi_max=255, saturation=10) == 0.0
    assert antsim.phi_linear(10, phi_max=255, saturation=10) == 255.0
    assert antsim.phi_linear(100, phi_max=255, saturation=10) == 255.0


def test_build_turn_kernel_allowed_values():
    turns = {0, 1, -1, 2, -2, 3, -3, 4}
    samples = [antsim.build_turn_kernel((0.360, 0.047, 0.008, 0.004)) for _ in range(1000)]
    assert set(samples).issubset(turns)


def test_ant_move_one_step():
    for d in range(8):
        a = antsim.Ant(10, 10, d)
        x0, y0 = a.x, a.y
        a.move_one_step()
        assert (a.x - x0, a.y - y0) == (int(antsim.STEP_X[d]), int(antsim.STEP_Y[d]))


def test_choose_follower_direction_forward_priority():
    pher = np.zeros((5, 5), dtype=float)
    x, y, d = 2, 2, 0
    fx, fy = x + int(antsim.STEP_X[d]), y + int(antsim.STEP_Y[d])
    pher[fy, fx] = 1.0

    nd = antsim.choose_follower_direction(x, y, d, pher, (0.360, 0.047, 0.008, 0.004))
    assert nd == d


def test_simulation_release_counts():
    sim = antsim.Simulation(grid_size=16, steps=10, max_ants=3, release_rate=1, nest=(8, 8))
    sim.run()
    assert sim.total_released == 3


def test_evaporation_nonnegative():
    sim = antsim.Simulation(grid_size=8, steps=1, max_ants=1, release_rate=1, nest=(4, 4))
    sim.deposit_amount = 1.0
    sim.evaporation = 5.0
    sim.step()
    assert sim.pheromone.min() >= 0.0
