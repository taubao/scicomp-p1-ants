"""Simple checks"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.antsim import (
    STEP_X,
    STEP_Y,
    Ant,
    Simulation,
    build_turn_kernel,
    choose_follower_direction,
    phi_linear,
)


# --- Constants ---
B = (0.360, 0.047, 0.008, 0.004)
TURNS = np.array([0, 1, -1, 2, -2, 3, -3, 4], dtype=int)
NEIGHBOR_STEPS = {(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)}


# --- Kernel sampling ---
def check_turn_kernel_sampling():
    B1, B2, B3, B4 = B
    p0 = 1.0 - (B1 + B2 + B3 + B4)
    p_expected = np.array([p0, B1 / 2, B1 / 2, B2 / 2, B2 / 2, B3 / 2, B3 / 2, B4], dtype=float)
    p_expected /= p_expected.sum()

    # Draw many samples to compare observed vs expected
    sample_count = 20000
    samples = np.array([build_turn_kernel(B) for _ in range(sample_count)], dtype=int)

    assert set(np.unique(samples)).issubset(set(TURNS))

    freq = np.array([(samples == t).mean() for t in TURNS], dtype=float)
    max_err = float(np.max(np.abs(freq - p_expected)))

    print("turns:     ", TURNS)
    print("expected p:", np.round(p_expected, 4), "sum=", round(float(p_expected.sum()), 6))
    print("observed f:", np.round(freq, 4))
    print("max_err:", round(max_err, 4))

    assert max_err < 0.01, "frequencies are too far from expected; kernel sampling may be wrong"
    print("OK: kernel sampling matches expected probabilities.")


# --- Direction updates ---
def check_direction_update():
    for d in range(8):
        for _ in range(200):
            turn = build_turn_kernel(B)
            new_dir = (d + turn) & 7
            assert 0 <= new_dir <= 7

            # Movement step should be one of the 8 neighbor moves
            dx = int(STEP_X[new_dir])
            dy = int(STEP_Y[new_dir])
            assert (dx, dy) in NEIGHBOR_STEPS

    print("OK: direction update + movement lookup work.")


def check_ant_class():
    for d in range(8):
        a = Ant(10, 10, d)
        x0, y0 = a.x, a.y
        a.move_one_step()
        assert (a.x - x0, a.y - y0) == (int(STEP_X[d]), int(STEP_Y[d]))
        assert a.in_bounds(256)

    # NW from corner should go out of bounds
    a = Ant(0, 0, 3)
    a.move_one_step()
    assert not a.in_bounds(256)

    print("OK: Ant class works.")


def check_ant_choose_direction():
    a = Ant(10, 10, direction=0)

    seen = set()
    for _ in range(500):
        a.choose_direction(B)
        assert 0 <= a.direction <= 7
        seen.add(a.direction)

    assert len(seen) > 1, "Direction never changes — kernel not applied"
    print("OK: Ant.choose_direction updates direction correctly.")


# --- Simulation basics ---
def check_simulation_runs():
    sim = Simulation(grid_size=32, steps=10, max_ants=5, release_rate=1, nest=(16, 16))
    sim.run()

    assert sim.total_released == 5
    assert 0 <= len(sim.ants) <= 5

    print("OK: Simulation run works.")


def check_full_simulation_class():
    sim = Simulation(grid_size=32, steps=1, max_ants=1, release_rate=1, nest=(16, 16))
    sim.deposit_amount = 8.0
    sim.evaporation = 1.0

    # Before step
    assert sim.total_released == 0
    assert len(sim.ants) == 0
    assert float(sim.pheromone.sum()) == 0.0

    # One step: release, deposit at nest, move, evaporate
    sim.step()

    assert sim.total_released == 1
    assert 0 <= len(sim.ants) <= 1

    nest_x, nest_y = sim.nest
    assert abs(float(sim.pheromone[nest_y, nest_x]) - 7.0) < 1e-9
    assert sim.pheromone.min() >= 0.0
    assert abs(float(sim.pheromone.sum()) - 7.0) < 1e-9

    print("OK: Simulation class release/direction/deposit/move/evaporate works.")


def check_sim_changes_direction():
    sim = Simulation(grid_size=32, steps=1, max_ants=1, release_rate=1, nest=(16, 16))
    sim.release_ants()
    sim.step()

    # Ant might be gone, so only check if still on grid
    if sim.ants:
        assert 0 <= sim.ants[0].direction <= 7
    print("OK: sim step updates direction.")


# --- Pheromone behaviors ---
def check_pheromone_bias_prefers_forward():
    n = 11
    pher = np.zeros((n, n), dtype=float)

    # Ant at center facing east; raise pheromone in the forward cell
    ax, ay = 5, 5
    forward_dir = 0
    fx = ax + int(STEP_X[forward_dir])
    fy = ay + int(STEP_Y[forward_dir])
    pher[fy, fx] = 10.0

    sample_count = 2000
    forward = 0
    for _ in range(sample_count):
        a = Ant(ax, ay, direction=0)
        a.choose_direction(B, pheromone=pher, alpha=0.8, cap=10.0)
        if a.direction == 0:
            forward += 1

    frac = forward / sample_count
    print("forward fraction:", round(frac, 3))
    assert frac > 0.75
    print("OK: pheromone bias increases probability of moving toward higher pheromone.")


def check_phi_linear():
    assert phi_linear(0, phi_max=255, saturation=10) == 0.0
    assert abs(phi_linear(5, phi_max=255, saturation=10) - 127.5) < 1e-9
    assert phi_linear(10, phi_max=255, saturation=10) == 255.0
    assert phi_linear(100, phi_max=255, saturation=10) == 255.0
    print("OK: phi_linear is linear then saturates.")


def check_fork_algo_v1():
    n = 9
    pher = np.zeros((n, n), dtype=float)

    # Ant at center facing east
    x, y, d = 4, 4, 0
    fdir = d
    ldir = (d + 1) & 7
    rdir = (d - 1) & 7

    fx, fy = x + int(STEP_X[fdir]), y + int(STEP_Y[fdir])
    lx, ly = x + int(STEP_X[ldir]), y + int(STEP_Y[ldir])
    rx, ry = x + int(STEP_X[rdir]), y + int(STEP_Y[rdir])

    # Case 1: true fork and forward continues
    pher[:] = 0
    pher[ly, lx] = 2
    pher[ry, rx] = 2
    pher[fy, fx] = 2
    nd = choose_follower_direction(x, y, d, pher, B)
    assert nd == fdir

    # Case 2: true fork and forward absent
    pher[:] = 0
    pher[ly, lx] = 5
    pher[ry, rx] = 2
    pher[fy, fx] = 0
    nd = choose_follower_direction(x, y, d, pher, B)
    assert nd == ldir

    print("OK: follower fork algorithm v1 behaves as expected.")


def check_lost_not_follow_off_trail():
    sim = Simulation(grid_size=32, steps=1, max_ants=1, release_rate=1, nest=(16, 16))
    sim.fidelity = 255

    sim.release_ants()
    sim.ants[0].is_follower = False

    # Pheromone at nest is 0 before deposit
    sim.step()
    if sim.ants:
        assert sim.ants[0].is_follower is False

    print("OK: lost ants do not become followers off-trail.")


def check_follow_distance_counter_runs():
    sim = Simulation(grid_size=32, steps=5, max_ants=5, release_rate=1, nest=(16, 16))
    sim.run()
    m = sim.mean_trail_follow_distance()
    assert m >= 0.0
    print("OK: mean_trail_follow_distance computes:", m)


# --- Runner ---
def run_all_checks():
    check_turn_kernel_sampling()
    check_direction_update()
    check_ant_class()
    check_ant_choose_direction()
    check_simulation_runs()
    check_full_simulation_class()
    check_sim_changes_direction()
    check_pheromone_bias_prefers_forward()
    check_phi_linear()
    check_fork_algo_v1()
    check_lost_not_follow_off_trail()
    check_follow_distance_counter_runs()


if __name__ == "__main__":
    run_all_checks()