"""Synthetic price generator and example signals for edge-decay demos.

All data here is synthetic and seeded. Nothing in this module touches the
network or any real market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_prices(
    n: int = 3000,
    phi: float = 0.9,
    alpha_sigma: float = 0.0015,
    noise_sigma: float = 0.01,
    price0: float = 100.0,
    seed: int = 0,
) -> pd.Series:
    """Prices with a hidden AR(1) "alpha" component plus i.i.d. noise.

    hidden_alpha[t] = phi * hidden_alpha[t-1] + eps[t]
    return[t]       = hidden_alpha[t] + noise[t]
    price[t]        = price[t-1] * (1 + return[t])

    `phi` controls how persistent (slow-decaying) the hidden alpha is.
    Set `alpha_sigma=0` for a pure-noise control series (no real edge).
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, alpha_sigma, n)
    hidden_alpha = np.zeros(n)
    hidden_alpha[0] = eps[0]
    for t in range(1, n):
        hidden_alpha[t] = phi * hidden_alpha[t - 1] + eps[t]

    noise = rng.normal(0.0, noise_sigma, n)
    returns = hidden_alpha + noise
    prices = price0 * np.cumprod(1.0 + returns)
    return pd.Series(prices, index=pd.RangeIndex(n), name="price")


def fast_alpha_signal(prices: pd.Series, k: float = 40.0) -> pd.Series:
    """Reacts to the most recent 1-bar return. Exploits alpha that decays fast.

    Uses only prices up to and including bar t (pct_change at t uses p[t-1], p[t]).
    """
    r = prices.pct_change().fillna(0.0)
    return (k * r).clip(-1.0, 1.0)


def slow_trend_signal(prices: pd.Series, window: int = 20, k: float = 60.0) -> pd.Series:
    """Reacts to a rolling mean of recent returns. Exploits persistent trends.

    The rolling window only looks backward, so it uses data up to and
    including bar t and introduces no lookahead.
    """
    r = prices.pct_change().fillna(0.0)
    trend = r.rolling(window, min_periods=1).mean()
    return (k * trend).clip(-1.0, 1.0)
