from typing import Tuple
import numpy as np


def simulate_two_state_markov_chain(
    t_end: float,
    dt: float,
    lambda_01: float,
    lambda_10: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a two state continuous time Markov chain on a discrete grid.

    States: 0 = calm, 1 = stressed.

    At each small time step dt we approximate transition probabilities as:
    p(0->1) ~ lambda_01 * dt
    p(1->0) ~ lambda_10 * dt
    """
    rng = np.random.default_rng(seed)
    num_steps = int(t_end / dt) + 1
    times = np.linspace(0.0, t_end, num_steps)

    states = np.zeros(num_steps, dtype=int)
    state = 0

    for i in range(1, num_steps):
        if state == 0:
            p = lambda_01 * dt
            if rng.random() < p:
                state = 1
        else:
            p = lambda_10 * dt
            if rng.random() < p:
                state = 0
        states[i] = state

    return times, states
