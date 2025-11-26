from typing import Dict, Tuple

import numpy as np
import pandas as pd


def summarize_paths(price_df: pd.DataFrame) -> Dict[str, float]:
    """
    Basic statistics on the final prices across all paths.
    """
    final_prices = price_df.iloc[-1].values
    return {
        "mean_final_price": float(final_prices.mean()),
        "std_final_price": float(final_prices.std()),
        "min_final_price": float(final_prices.min()),
        "max_final_price": float(final_prices.max()),
    }


def compute_percentiles(price_df: pd.DataFrame, percentiles: Tuple[float, ...] = (5.0, 50.0, 95.0)) -> pd.DataFrame:
    """
    Compute percentiles at each time point across all paths.
    """
    perc_values = np.percentile(price_df.values, q=list(percentiles), axis=1)
    perc_df = pd.DataFrame(
        perc_values.T,
        index=price_df.index,
        columns=[f"p{int(p)}" for p in percentiles],
    )
    return perc_df
