from pathlib import Path

import matplotlib.pyplot as plt

from src.config import RegimeConfig, HestonConfig, JumpConfig, SimulationConfig
from src.simulator import run_monte_carlo_simulation
from src.analytics import summarize_paths, compute_percentiles


def main() -> None:
    regime_cfg = RegimeConfig()
    heston_cfg = HestonConfig()
    jump_cfg = JumpConfig()
    sim_cfg = SimulationConfig()

    print("Running Monte Carlo simulation for power prices")
    price_df, var_df = run_monte_carlo_simulation(
        regime_cfg=regime_cfg,
        heston_cfg=heston_cfg,
        jump_cfg=jump_cfg,
        sim_cfg=sim_cfg,
    )

    stats = summarize_paths(price_df)
    print("Summary of final prices:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    perc_df = compute_percentiles(price_df)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    price_df.to_csv(out_dir / "prices.csv")
    var_df.to_csv(out_dir / "variances.csv")
    perc_df.to_csv(out_dir / "price_percentiles.csv")

    # Plot a few sample paths and percentile bands
    fig, ax = plt.subplots(figsize=(10, 5))

    sample_cols = list(price_df.columns[:10])
    ax.plot(price_df.index, price_df[sample_cols], alpha=0.3)

    ax.plot(perc_df.index, perc_df["p50"], linewidth=2, label="Median")
    ax.fill_between(
        perc_df.index,
        perc_df["p5"],
        perc_df["p95"],
        alpha=0.2,
        label="5 to 95 percent band",
    )

    ax.set_title("Simulated power price paths")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.legend(loc="best")

    fig.tight_layout()
    fig_path = out_dir / "price_paths.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"Saved prices, variances and percentiles to {out_dir.resolve()}")
    print(f"Saved plot to {fig_path.resolve()}")


if __name__ == "__main__":
    main()
