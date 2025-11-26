from typing import Tuple
import numpy as np

from .config import RegimeConfig, HestonConfig, JumpConfig


def simulate_heston_jump_diffusion_path(
    times: np.ndarray,
    regimes: np.ndarray,
    regime_cfg: RegimeConfig,
    heston_cfg: HestonConfig,
    jump_cfg: JumpConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single price and variance path with Heston dynamics and jumps.

    SDE in log price terms (simplified):
        dS/S = (mu_regime) dt + sqrt(v_regime) dW1 + J dN

    Variance:
        dv = kappa (theta - v) dt + sigma_v sqrt(v) dW2

    with Corr(dW1, dW2) = rho.
    """
    num_steps = len(times)
    dt_array = np.diff(times)
    dt_array = np.concatenate(([dt_array[0]], dt_array))  # same length as times

    s = np.empty(num_steps)
    v = np.empty(num_steps)

    s[0] = heston_cfg.s0
    v[0] = heston_cfg.v0

    for i in range(1, num_steps):
        dt = dt_array[i]
        state = regimes[i]

        if state == 0:
            mu = regime_cfg.calm_mu
            vol_mult = regime_cfg.calm_vol_multiplier
        else:
            mu = regime_cfg.stressed_mu
            vol_mult = regime_cfg.stressed_vol_multiplier

        z1 = rng.normal()
        z2 = rng.normal()

        # Correlated Brownian motions
        dW1 = np.sqrt(dt) * z1
        dW2 = np.sqrt(dt) * (regime_cfg.calm_mu * 0.0)  # dummy init
        dW2 = np.sqrt(dt) * (heston_cfg.rho * z1 + np.sqrt(1.0 - heston_cfg.rho**2) * z2)

        v_prev = max(v[i - 1], 1e-8)
        dv = (
            heston_cfg.kappa * (heston_cfg.theta - v_prev) * dt
            + heston_cfg.sigma_v * np.sqrt(v_prev) * dW2
        )
        v_i = max(v_prev + dv, 1e-8)

        lambda_jump = jump_cfg.intensity
        prob_jump = lambda_jump * dt
        has_jump = rng.random() < prob_jump
        jump_term = 0.0
        if has_jump:
            j = rng.normal(jump_cfg.mean_jump, jump_cfg.std_jump)
            jump_term = j

        vol_effective = vol_mult * np.sqrt(v_i)
        ds_over_s = mu * dt + vol_effective * dW1 + jump_term

        s[i] = s[i - 1] * np.exp(ds_over_s)
        v[i] = v_i

    return s, v
