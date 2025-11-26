from typing import Tuple

import numpy as np
import pandas as pd

from .config import RegimeConfig, HestonConfig, JumpConfig, SimulationConfig
from .regime_switch import simulate_two_state_markov_chain
from .heston_jump_diffusion import simulate_heston_jump_diffusion_path


def run_monte_carlo_simulation(
    regime_cfg: RegimeConfig,
    heston_cfg: HestonConfig,
    jump_cfg: JumpConfig,
    sim_cfg: SimulationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Monte Carlo simulation for price and variance paths.

    Returns two DataFrames:
      - prices: index = times, columns = path_0, path_1, ...
      - variances: same shape as prices.
    """
    times, regimes = simulate_two_state_markov_chain(
        t_end=sim_cfg.t_end,
        dt=sim_cfg.dt,
        lambda_01=regime_cfg.lambda_01,
        lambda_10=regime_cfg.lambda_10,
        seed=sim_cfg.seed,
    )

    rng = np.random.default_rng(sim_cfg.seed)

    num_steps = len(times)
    num_paths = sim_cfg.num_paths

    prices = np.empty((num_steps, num_paths))
    variances = np.empty((num_steps, num_paths))

    for p in range(num_paths):
        s_path, v_path = simulate_heston_jump_diffusion_path(
            times=times,
            regimes=regimes,
            regime_cfg=regime_cfg,
            heston_cfg=heston_cfg,
            jump_cfg=jump_cfg,
            rng=rng,
        )
        prices[:, p] = s_path
        variances[:, p] = v_path

    index = pd.Index(times, name="time")
    price_df = pd.DataFrame(
        prices,
        index=index,
        columns=[f"path_{i}" for i in range(num_paths)],
    )
    var_df = pd.DataFrame(
        variances,
        index=index,
        columns=[f"path_{i}" for i in range(num_paths)],
    )

    return price_df, var_df
