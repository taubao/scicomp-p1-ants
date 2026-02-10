"""Ant trail simulation primitives and runtime helpers."""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# 8-connected grid offsets for directions 0..7.
STEP_X = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=int)
STEP_Y = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=int)


def phi_linear(pheromone, phi_max=255.0, saturation=10.0):
    """Map pheromone to intensity with linear ramp, then saturation."""
    if pheromone <= 0:
        return 0.0
    if pheromone >= saturation:
        return float(phi_max)
    # Scale proportionally within the linear range.
    return float(phi_max) * (float(pheromone) / float(saturation))


def choose_follower_direction(x, y, direction, pheromone, turn_kernel, eps=1e-9):
    """Select a follower direction for forks."""
    n = pheromone.shape[0]

    forward_dir = direction
    left_dir = (direction + 1) & 7
    right_dir = (direction - 1) & 7

    fx, fy = x + int(STEP_X[forward_dir]), y + int(STEP_Y[forward_dir])
    lx, ly = x + int(STEP_X[left_dir]), y + int(STEP_Y[left_dir])
    rx, ry = x + int(STEP_X[right_dir]), y + int(STEP_Y[right_dir])

    # If any neighbor is out of bounds, keep moving forward.
    if not (0 <= fx < n and 0 <= fy < n and 0 <= lx < n and 0 <= ly < n and 0 <= rx < n and 0 <= ry < n):
        return forward_dir

    pher_forward = float(pheromone[fy, fx])
    pher_left = float(pheromone[ly, lx])
    pher_right = float(pheromone[ry, rx])

    # Forward has priority if it is on-trail.
    if pher_forward > 0:
        return forward_dir

    # If left and right are tied, pick a random turn.
    if abs(pher_left - pher_right) <= eps:
        turn = build_turn_kernel(turn_kernel)
        return (direction + int(turn)) & 7
    return left_dir if pher_left > pher_right else right_dir


def sense_forward_left_right(x, y, direction, pheromone):
    """Read pheromone in the forward/left/right cells."""
    n = pheromone.shape[0]
    forward_dir = direction
    left_dir = (direction + 1) & 7
    right_dir = (direction - 1) & 7

    fx, fy = x + int(STEP_X[forward_dir]), y + int(STEP_Y[forward_dir])
    lx, ly = x + int(STEP_X[left_dir]), y + int(STEP_Y[left_dir])
    rx, ry = x + int(STEP_X[right_dir]), y + int(STEP_Y[right_dir])

    # Treat out-of-bounds as zero pheromone.
    pher_forward = pheromone[fy, fx] if (0 <= fx < n and 0 <= fy < n) else 0.0
    pher_left = pheromone[ly, lx] if (0 <= lx < n and 0 <= ly < n) else 0.0
    pher_right = pheromone[ry, rx] if (0 <= rx < n and 0 <= ry < n) else 0.0
    return float(pher_forward), float(pher_left), float(pher_right)

def build_turn_kernel(turn_kernel):
    """Determine the probabilities of the 8 possible turns."""
    B1, B2, B3, B4 = turn_kernel
    p0 = 1.0 - (B1 + B2 + B3 + B4)
    assert p0 >= 0

    # Probability for left/right turns is split evenly.
    p = np.array([p0, B1 / 2, B1 / 2, B2 / 2, B2 / 2, B3 / 2, B3 / 2, B4], dtype=float)
    p /= p.sum()

    # Possible turns, based on direction and number of 45-degree steps.
    turns = np.array([0, 1, -1, 2, -2, 3, -3, 4], dtype=int)
    return int(rng.choice(turns, p=p))

class Ant:
    """Single ant state: position, direction, and follower status."""

    def __init__(self, x, y, direction):
        """Initialize a new ant at (x, y) with a direction."""
        self.x = int(x)
        self.y = int(y)
        # 0..7 to stay within the 8-direction encoding.
        self.direction = int(direction) & 7
        self.is_follower = False
        self.follow_run_length = 0

    def choose_direction(self, turn_kernel, pheromone=None, alpha=0.6, cap=10.0):
        """Choose a new direction from the kernel."""
        B1, B2, B3, B4 = turn_kernel
        p0 = 1.0 - (B1 + B2 + B3 + B4)
        assert p0 >= 0

        turns = np.array([0, 1, -1, 2, -2, 3, -3, 4], dtype=int)
        p = np.array([p0, B1 / 2, B1 / 2, B2 / 2, B2 / 2, B3 / 2, B3 / 2, B4], dtype=float)
        p /= p.sum()

        # Without pheromone, choose randomly.
        if pheromone is None:
            turn = int(rng.choice(turns, p=p))
            self.direction = (self.direction + turn) & 7
            return

        n = pheromone.shape[0]
        
        # Next step: Start from kernel probabilities and reweight based on pheromone.
        w = p.copy()

        for i, turn_i in enumerate(turns):
            new_dir = (self.direction + int(turn_i)) & 7
            nx = self.x + int(STEP_X[new_dir])
            ny = self.y + int(STEP_Y[new_dir])

            # Read pheromone in candidate cell, using 0 for out-of-bounds.
            if 0 <= nx < n and 0 <= ny < n:
                c_val = float(pheromone[ny, nx])
            else:
                c_val = 0.0

            if c_val > cap:
                c_val = cap

            # More bias toward higher pheromone cells.
            w[i] *= np.exp(alpha * c_val)

        # Normalize to get probabilities and get a turn.
        w /= w.sum()
        turn = int(rng.choice(turns, p=w))
        self.direction = (self.direction + turn) & 7

    def move_one_step(self):
        """Advance the ant one grid step in its current direction."""
        self.x += int(STEP_X[self.direction])
        self.y += int(STEP_Y[self.direction])

    def in_bounds(self, n):
        """Return True if the ant is inside an n-by-n grid."""
        return 0 <= self.x < n and 0 <= self.y < n


class Simulation:
    """Ant simulation logic."""
    # Lower default for speed for testing, but overridden in main.
    def __init__(self, grid_size=256, steps=100, max_ants=500, release_rate=1, nest=(128, 128)):
        """Configure a simulation with grid and release parameters."""
        self.grid_size = int(grid_size)
        self.steps = int(steps)
        self.max_ants = int(max_ants)
        self.release_rate = int(release_rate)
        self.nest = (int(nest[0]), int(nest[1]))

        self.ants = []
        self.total_released = 0

        # Turning kernel and pheromone parameters.
        self.turn_kernel = (0.360, 0.047, 0.008, 0.004)

        self.pheromone = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.deposit_amount = 8.0
        self.evaporation = 1.0
        self.fidelity = 251
        # Tracking follower run statistics.
        self.total_follow_run_length = 0
        self.total_follow_runs = 0
        self.debug_followers = False
        self.last_followers = 0
        self.last_lost = 0

    def follower_counts(self):
        """Return (followers, lost) counts among current ants."""
        followers = sum(1 for a in self.ants if a.is_follower)
        lost = len(self.ants) - followers
        return followers, lost

    def mean_trail_follow_distance(self):
        """Return the mean follower run length across completed runs."""
        if self.total_follow_runs == 0:
            return 0.0
        return self.total_follow_run_length / self.total_follow_runs

    def release_ants(self):
        """Release new ants at the nest up to the max population."""
        if self.total_released >= self.max_ants:
            return

        # Release up to the max number of ants.
        to_release = min(self.release_rate, self.max_ants - self.total_released)
        x0, y0 = self.nest

        for _ in range(to_release):
            d0 = int(rng.integers(0, 8))
            self.ants.append(Ant(x0, y0, d0))
            self.total_released += 1

    def deposit(self):
        """Deposit pheromone at each ant's current position."""
        for a in self.ants:
            if 0 <= a.x < self.grid_size and 0 <= a.y < self.grid_size:
                self.pheromone[a.y, a.x] += self.deposit_amount

    def evaporate(self):
        """Apply linear pheromone evaporation and clamp at zero."""
        self.pheromone -= self.evaporation
        self.pheromone[self.pheromone < 0] = 0.0

    def step(self):
        """Advance the simulation by one time step."""
        # Release new ants first so they participate in this step.
        self.release_ants()

        for a in self.ants:
            pheromone_here = float(self.pheromone[a.y, a.x])
            # Fidelity controls probability of staying in follower state.
            stay_prob = self.fidelity / 256.0
            pher_forward, pher_left, pher_right = sense_forward_left_right(
                a.x, a.y, a.direction, self.pheromone
            )
            trail_seen = (pher_forward > 0) or (pher_left > 0) or (pher_right > 0)

            if a.is_follower:
                # Followers can drop if local signal is weak or by chance.
                if pheromone_here <= 0 or rng.random() >= stay_prob:
                    a.is_follower = False
            else:
                # Lost ants become followers if they detect a trail.
                if trail_seen and rng.random() < stay_prob:
                    a.is_follower = True

        for a in self.ants:
            # Followers use the fork rule, explorers use the kernel.
            if a.is_follower:
                a.direction = choose_follower_direction(
                    a.x, a.y, a.direction, self.pheromone, self.turn_kernel
                )
            else:
                a.choose_direction(self.turn_kernel)

        for a in self.ants:
            pheromone_here = self.pheromone[a.y, a.x]
            on_trail = a.is_follower and pheromone_here > 0
            if on_trail:
                a.follow_run_length += 1
            elif a.follow_run_length > 0:
                # End a follower run when an ant is no longer a follower.
                self.total_follow_run_length += a.follow_run_length
                self.total_follow_runs += 1
                a.follow_run_length = 0

        if self.debug_followers:
            followers, lost = self.follower_counts()
            print(f"followers={followers} lost={lost}")

        # Deposit pheromone before ants move to the next cell.
        self.deposit()

        # Move ants and remove those that exit the grid.
        on_grid = []
        n = self.grid_size
        for a in self.ants:
            a.move_one_step()
            if a.in_bounds(n):
                on_grid.append(a)
            elif a.follow_run_length > 0:
                # Close out run lengths for ants that leave the grid.
                self.total_follow_run_length += a.follow_run_length
                self.total_follow_runs += 1
                a.follow_run_length = 0
        self.ants = on_grid

        # Apply evaporation after all moves and deposits.
        self.evaporate()

    def run(self):
        """Run the simulation for the number of steps."""
        for _ in range(self.steps):
            self.step()
        for a in self.ants:
            if a.follow_run_length > 0:
                # Close out any runs that persist to the end.
                self.total_follow_run_length += a.follow_run_length
                self.total_follow_runs += 1
                a.follow_run_length = 0
        self.last_followers, self.last_lost = self.follower_counts()

    def save_image(self, out_path, vmax=None):
        """Save a grayscale pheromone image with optional ant overlay."""
        img = self.pheromone
        followers, lost = self.follower_counts()
        mean_steps = self.mean_trail_follow_distance()
        if vmax is None:
            # Use a high percentile for contrast while avoiding outliers.
            vmax = float(np.percentile(img, 99.5)) if img.size else 1.0
        if vmax <= 0:
            vmax = 1.0

        plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(img, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=vmax)
        if self.ants:
            # Overlay active ant positions.
            xs = [a.x for a in self.ants]
            ys = [a.y for a in self.ants]
            plt.scatter(xs, ys, s=2, c="black", linewidths=0)

        plt.title(
            f"fidelity={self.fidelity} followers={followers} lost={lost} mean_follow_steps={mean_steps:.2f}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()


def run_and_save_image():
    """Run three fidelities and save their pheromone images."""
    sims = []
    for fidelity_value in (247, 251, 255):
        sim = Simulation(steps=1500, max_ants=1200, release_rate=1)
        sim.fidelity = fidelity_value
        sim.run()
        sims.append(sim)

    # Share a common color scale across all outputs. Finds the 99.5th percentile of pheromone values
    vmax = max(float(np.percentile(sim.pheromone, 99.5)) for sim in sims)
    if vmax <= 0:
        vmax = 1.0

    for sim in sims:
        sim.save_image(f"output/sim_fidelity_{sim.fidelity}.png", vmax=vmax)