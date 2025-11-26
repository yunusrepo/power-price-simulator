from dataclasses import dataclass


@dataclass
class RegimeConfig:
    # Transition intensities per unit time
    lambda_01: float = 0.1  # calm -> stressed
    lambda_10: float = 0.3  # stressed -> calm

    calm_mu: float = 0.0
    stressed_mu: float = 0.0

    calm_vol_multiplier: float = 1.0
    stressed_vol_multiplier: float = 2.5


@dataclass
class HestonConfig:
    kappa: float = 2.0      # mean reversion speed
    theta: float = 0.04     # long run variance
    sigma_v: float = 0.5    # vol of vol
    rho: float = -0.5       # correlation between price and variance shocks
    v0: float = 0.04        # initial variance
    s0: float = 100.0       # initial price


@dataclass
class JumpConfig:
    intensity: float = 3.0      # jumps per year
    mean_jump: float = 0.08     # mean jump size (log space)
    std_jump: float = 0.25      # jump size volatility


@dataclass
class SimulationConfig:
    t_end: float = 1.0          # years
    dt: float = 1.0 / (24 * 365)  # approx hourly steps in years
    num_paths: int = 200
    seed: int = 42
