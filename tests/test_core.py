import numpy as np
import pandas as pd
import pytest

from edgedecay.core import backtest, breakeven_cost_bps, decay_curve, half_life


@pytest.fixture
def clean_prices():
    # Each bar is exactly +10% so pct_change is a clean 0.1 for every step.
    return pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])


def test_delay_shifts_executed_positions_exactly(clean_prices):
    positions = pd.Series([1.0, 0.0, -1.0, 1.0, 0.0])
    returns = clean_prices.pct_change().fillna(0.0)

    for delay in (0, 1, 2):
        net = backtest(clean_prices, positions, delay=delay, cost_bps=0.0)
        executed = positions.shift(1 + delay).fillna(0.0)
        expected = executed * returns
        np.testing.assert_allclose(net.to_numpy(), expected.to_numpy())


def test_cost_reduces_returns_by_the_right_amount(clean_prices):
    positions = pd.Series([1.0, 0.0, -1.0, 1.0, 0.0])
    net = backtest(clean_prices, positions, delay=0, cost_bps=50.0)
    # hand-computed: executed=[0,1,0,-1,1], turnover=[0,1,1,1,2],
    # cost=0.005*turnover, returns=[0,.1,.1,.1,.1]
    expected = [0.0, 0.095, -0.005, -0.105, 0.09]
    np.testing.assert_allclose(net.to_numpy(), expected, atol=1e-12)


def test_cost_reduces_returns_at_nonzero_delay(clean_prices):
    positions = pd.Series([1.0, 0.0, -1.0, 1.0, 0.0])
    net = backtest(clean_prices, positions, delay=1, cost_bps=50.0)
    # executed=[0,0,1,0,-1], turnover=[0,0,1,1,1]
    expected = [0.0, 0.0, 0.095, -0.005, -0.105]
    np.testing.assert_allclose(net.to_numpy(), expected, atol=1e-12)


def test_breakeven_cost_zeroes_out_net_sharpe():
    rng = np.random.default_rng(42)
    prices = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, 500)))

    def signal_fn(p):
        r = p.pct_change().fillna(0.0)
        return (20.0 * r.rolling(5, min_periods=1).mean()).clip(-1.0, 1.0)

    be = breakeven_cost_bps(prices, signal_fn, delay=0)
    assert np.isfinite(be)

    curve = decay_curve(prices, signal_fn, delays=[0], cost_bps=be)
    assert abs(curve["sharpe"].iloc[0]) < 1e-6


def test_breakeven_is_inf_when_no_turnover():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])

    def flat_signal(p):
        return pd.Series(0.0, index=p.index)

    be = breakeven_cost_bps(prices, flat_signal, delay=0)
    assert be == float("inf")


def test_half_life_none_when_delay0_sharpe_not_positive():
    curve = pd.DataFrame({"sharpe": [-0.5, -0.4, -0.3]}, index=[0, 1, 2])
    assert half_life(curve) is None


def test_half_life_interpolates_between_grid_points():
    # sharpe halves exactly at delay=1.5 by linear interpolation
    curve = pd.DataFrame({"sharpe": [2.0, 1.0, 0.0]}, index=[0, 1, 2])
    assert half_life(curve) == pytest.approx(1.0)

    curve2 = pd.DataFrame({"sharpe": [2.0, 1.5, 0.5]}, index=[0, 1, 2])
    # target = 1.0, between index1 (1.5) and index2 (0.5): frac=(1.5-1.0)/(1.5-0.5)=0.5 -> delay 1.5
    assert half_life(curve2) == pytest.approx(1.5)


def test_half_life_none_when_never_reached():
    curve = pd.DataFrame({"sharpe": [2.0, 1.9, 1.8]}, index=[0, 1, 2])
    assert half_life(curve) is None
