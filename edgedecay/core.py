"""Measure how a trading signal's edge decays with execution delay and cost.

Research-only. No trading, no network calls, no live data.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

SignalFn = Callable[[pd.Series], pd.Series]


def _executed_position(positions: pd.Series, delay: int) -> pd.Series:
    """Position actually held at each bar, after shifting by 1 + delay bars."""
    return positions.shift(1 + delay).fillna(0.0)


def _active(series: pd.Series, delay: int) -> pd.Series:
    """Drop the `1 + delay` warm-up bars during which no position is held yet."""
    return series.iloc[1 + delay :]


def backtest(
    prices: pd.Series,
    positions: pd.Series,
    delay: int = 0,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Per-bar net returns of holding `positions`, executed `delay` bars late.

    `positions[t]` is assumed to be computed using data up to and including
    bar t, and is applied to the bar t+1 return when delay=0 (i.e. shifted by
    1 bar). A positive `delay` pushes execution further into the future
    (shift by 1 + delay). Transaction cost is charged on every change in the
    executed position: cost_bps/1e4 * |executed_pos[t] - executed_pos[t-1]|.

    `delay` must be a non-negative integer: negative delay would execute on
    or before the bar the signal was computed from, i.e. lookahead.
    """
    if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)):
        raise ValueError(f"delay must be an integer, got {delay!r}")
    if delay < 0:
        raise ValueError(f"delay must be non-negative, got {delay}")

    returns = prices.pct_change().fillna(0.0)
    executed = _executed_position(positions, delay)
    prev = executed.shift(1).fillna(0.0)
    turnover = (executed - prev).abs()
    cost = (cost_bps / 1e4) * turnover
    net = executed * returns - cost
    net.name = "net_return"
    return net


def decay_curve(
    prices: pd.Series,
    signal_fn: SignalFn,
    delays=range(0, 11),
    cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Sharpe / total return / turnover / t-stat of a signal across a grid of delays.

    Sharpe, turnover, and the t-stat are computed on the "active" bars only
    (those at or after `1 + delay`, once a position is actually held), so a
    larger delay's longer warm-up of forced-zero bars doesn't dilute the
    stats. `total_return` is a full-sample compounding and is unaffected by
    the warm-up bars, since their net return is exactly 0.

    The `t_stat` column is the t-statistic of the per-bar (non-annualized)
    mean active return, `sharpe_per_bar * sqrt(n_active)` — a rough
    significance check independent of `periods_per_year`.
    """
    positions = signal_fn(prices)
    rows = []
    for d in delays:
        net = backtest(prices, positions, delay=d, cost_bps=cost_bps)
        active = _active(net, d)
        n_active = len(active)
        mean = active.mean()
        std = active.std()
        no_edge = n_active == 0 or std == 0 or np.isnan(std)
        sharpe = 0.0 if no_edge else (mean / std) * np.sqrt(periods_per_year)
        t_stat = 0.0 if no_edge else (mean / std) * np.sqrt(n_active)
        total_return = float((1.0 + net).prod() - 1.0)

        executed = _executed_position(positions, d)
        prev = executed.shift(1).fillna(0.0)
        turnover = float(_active((executed - prev).abs(), d).mean())

        rows.append(
            {
                "delay": d,
                "sharpe": float(sharpe),
                "total_return": total_return,
                "turnover": turnover,
                "t_stat": float(t_stat),
            }
        )
    return pd.DataFrame(rows).set_index("delay")


def half_life(curve: pd.DataFrame) -> Optional[float]:
    """Delay at which Sharpe first crosses half of the delay-0 Sharpe.

    This is a *first-crossing* estimate on the given delay grid, linearly
    interpolated between the two grid points that straddle the crossing —
    it is not a monotonic decay-rate fit, so a curve that dips below the
    target and later recovers will still report the first dip. Grid points
    with a NaN Sharpe are skipped (interpolation uses the last finite point
    before them). Returns None if the delay-0 Sharpe is non-positive
    (nothing to decay from) or if no finite point on the grid reaches the
    target.
    """
    sharpes = curve["sharpe"].to_numpy(dtype=float)
    delays = curve.index.to_numpy(dtype=float)
    s0 = sharpes[0]
    if not np.isfinite(s0) or s0 <= 0:
        return None

    target = s0 / 2.0
    prev_delay, prev_sharpe = delays[0], s0
    for i in range(1, len(delays)):
        s_cur = sharpes[i]
        if not np.isfinite(s_cur):
            continue
        if s_cur <= target:
            if prev_sharpe == s_cur:
                return float(delays[i])
            frac = (prev_sharpe - target) / (prev_sharpe - s_cur)
            return float(prev_delay + frac * (delays[i] - prev_delay))
        prev_delay, prev_sharpe = delays[i], s_cur
    return None


def breakeven_cost_bps(
    prices: pd.Series,
    signal_fn: SignalFn,
    delay: int = 0,
) -> Optional[float]:
    """Cost (in bps) at which net Sharpe (equivalently, net mean return) hits 0.

    Closed form: mean(gross_return) == cost_frac * mean(turnover), so
    cost_frac = mean(gross_return) / mean(turnover), reported in bps. Stats
    are computed on active bars only (see `decay_curve`). Returns inf if the
    strategy never trades (zero turnover), and None if the gross mean return
    is non-positive: there is no edge to spend on costs, so there is no
    breakeven cost to report.
    """
    positions = signal_fn(prices)
    gross = _active(backtest(prices, positions, delay=delay, cost_bps=0.0), delay)

    executed = _executed_position(positions, delay)
    prev = executed.shift(1).fillna(0.0)
    turnover = _active((executed - prev).abs(), delay)
    mean_turnover = turnover.mean()

    if mean_turnover == 0:
        return float("inf")

    mean_gross = gross.mean()
    if not (mean_gross > 0):
        return None

    return float(mean_gross / mean_turnover * 1e4)
