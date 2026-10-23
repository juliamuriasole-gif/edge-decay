from functools import partial

import numpy as np

from edgedecay.core import decay_curve, half_life
from edgedecay.synthetic import fast_alpha_signal, generate_prices, slow_trend_signal

FAST_PRICES = generate_prices(n=6000, phi=0.15, alpha_sigma=0.008, noise_sigma=0.01, seed=1)
SLOW_PRICES = generate_prices(n=6000, phi=0.95, alpha_sigma=0.0012, noise_sigma=0.01, seed=2)
NOISE_PRICES = generate_prices(n=6000, phi=0.9, alpha_sigma=0.0, noise_sigma=0.01, seed=3)

FAST_SIGNAL = partial(fast_alpha_signal, k=15.0)
SLOW_SIGNAL = partial(slow_trend_signal, window=20, k=20.0)


def test_fast_signal_has_shorter_half_life_than_slow_signal():
    delays = range(0, 41)
    fast_curve = decay_curve(FAST_PRICES, FAST_SIGNAL, delays=delays)
    slow_curve = decay_curve(SLOW_PRICES, SLOW_SIGNAL, delays=delays)

    fast_hl = half_life(fast_curve)
    slow_hl = half_life(slow_curve)

    assert fast_hl is not None
    assert slow_hl is not None
    assert fast_hl < slow_hl


def test_noise_control_has_near_zero_sharpe_at_all_delays():
    delays = range(0, 21)
    curve = decay_curve(NOISE_PRICES, SLOW_SIGNAL, delays=delays)
    # Loose statistical bound: with n=6000 bars the Sharpe estimator has
    # std ~ sqrt(252/6000) ~ 0.2, so |sharpe| < 1.0 is a generous ~5-sigma band.
    assert (curve["sharpe"].abs() < 1.0).all()


def test_no_lookahead_truncating_future_prices_does_not_change_past_positions():
    for signal_fn in (FAST_SIGNAL, SLOW_SIGNAL):
        full_positions = signal_fn(FAST_PRICES)
        cutoff = len(FAST_PRICES) // 2
        truncated_positions = signal_fn(FAST_PRICES.iloc[:cutoff])
        np.testing.assert_allclose(
            truncated_positions.to_numpy(),
            full_positions.iloc[:cutoff].to_numpy(),
        )
