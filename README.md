# edgedecay

A small research tool for a question copy-trading bots ask informally all
the time: if you execute a signal `k` bars late, and pay transaction costs,
how much of its edge is left? Copy-trading tools often replay a wallet's
trades "as if copied 2-4 seconds late" to sanity-check whether an observed
edge is real alpha or just an artifact of acting first. This package
generalizes that check into a small, reusable backtesting utility: give it
any causal signal function and a price series, and it measures how Sharpe
ratio, total return, and turnover change as you delay execution bar by bar,
and reports the transaction cost at which the edge breaks even.

This is a research tool only. It does not place trades, does not call any
exchange or data API, and everything it runs on by default is synthetic,
seeded data. It is not investment advice.

## Install

```
python -m venv .venv
./.venv/bin/pip install -e .[test]
```

## Usage

A signal function takes a `pd.Series` of prices and returns a `pd.Series` of
positions in `[-1, 1]`, using only data up to and including each bar (no
lookahead). `backtest` shifts that position by `1 + delay` bars before
applying it to returns, and charges `cost_bps` on every change in the
executed position:

```python
import pandas as pd
from edgedecay.core import decay_curve, half_life, breakeven_cost_bps

def my_signal(prices: pd.Series) -> pd.Series:
    r = prices.pct_change().fillna(0.0)
    return (20.0 * r.rolling(20, min_periods=1).mean()).clip(-1.0, 1.0)

curve = decay_curve(prices, my_signal, delays=range(0, 11), cost_bps=1.0)
print(curve)
print("half-life:", half_life(curve))
print("breakeven cost (bps):", breakeven_cost_bps(prices, my_signal))
```

`decay_curve` takes `periods_per_year` (default `252`, i.e. daily bars) to
annualize Sharpe correctly for your bar frequency — e.g. pass `252 * 390`
for 1-minute equity bars, or `365 * 24` for hourly crypto bars. It also
reports a `t_stat` column: the t-statistic of the per-bar mean return on
active bars, independent of `periods_per_year`, used as a significance gate
(see below).

`breakeven_cost_bps` returns `None` when the gross mean return is
non-positive — there's no edge to spend on costs, so there's no breakeven
cost to report.

Run the bundled demo gallery (three synthetic scenarios: a signal that
exploits a fast-decaying hidden alpha, one that exploits a persistent trend,
and a pure-noise control) and write a report:

```
./.venv/bin/python -m edgedecay --output-dir report
```

This writes `report/report.md` (a table per strategy: Sharpe by delay,
half-life, breakeven cost) and `report/decay.png` (Sharpe vs. delay, all
three strategies overlaid). `--output-dir` defaults to `report` in the
current directory; `--periods-per-year` (default `252`) is passed through to
`decay_curve`. If the output directory can't be created or written to, the
CLI prints a one-line error to stderr and exits with status 2 instead of
raising.

### Significance gate

Half-life and breakeven cost are only meaningful if there's a statistically
significant edge at delay 0 in the first place — otherwise they're just
describing noise. The CLI and report gate on the delay-0 t-stat: by default,
if `t < 2.0` (`--min-t`), half-life and breakeven are reported as
`n/a (no significant edge, t=...)` instead of a number. The raw decay curve
(Sharpe/return/turnover by delay) is always shown regardless of the gate.

## Demo output

Actual output of `./.venv/bin/python examples/demo.py`:

```
[fast] fast-decaying hidden alpha (phi=0.15) + 1-bar-return signal
  sharpe(delay=0)=0.769  sharpe(delay=30)=-0.152  half_life=0.51 bars  breakeven_cost=5.62 bps
[slow] persistent hidden alpha (phi=0.95) + 20-bar trend signal
  sharpe(delay=0)=3.438  sharpe(delay=30)=0.572  half_life=14.45 bars  breakeven_cost=155.98 bps
[noise] pure noise (no hidden alpha) + the same 20-bar trend signal, as a control
  sharpe(delay=0)=0.101  sharpe(delay=30)=-0.201  half_life=n/a (no significant edge, t=0.49)  breakeven_cost=n/a (no significant edge, t=0.49)
```

The story this is meant to illustrate: the "fast" strategy's edge is
basically gone after a single bar of delay (half-life 0.51 bars) and it
breaks even at a thin 5.6 bps of cost, the "slow" trend strategy keeps a
real chunk of its Sharpe out to 30 bars of delay (half-life ~14.5 bars) and
tolerates ~156 bps of cost, and the "noise" control — the same trend signal
run on a price series with no injected alpha — has a delay-0 t-stat of only
0.49, well below the significance threshold, so its half-life and breakeven
cost are correctly reported as n/a rather than as numbers describing noise.

## Limitations

- All data is synthetic (a seeded AR(1) hidden-alpha process plus noise).
  Nothing here has been validated against real market data, and results on
  real data will look very different.
- Sharpe ratios on a single finite sample are noisy estimators; the
  per-delay numbers above bounce around even where there is no true signal
  (see the "noise" control). Don't read a single run's exact numbers as
  precise — look at the overall shape of the decay curve and re-run with
  more bars or multiple seeds if you need a tighter estimate.
- `half_life` and `breakeven_cost_bps` are computed from a specific grid of
  delays and a specific cost model (linear in turnover); they're estimates
  conditioned on those choices, not universal constants of a strategy. The
  significance gate only checks delay 0 — it doesn't vouch for every point
  on the curve.
- `periods_per_year` defaults to 252 (daily bars). If your bars are a
  different frequency (minutes, seconds, ticks), pass the correct value or
  annualized Sharpe will be off by orders of magnitude.
- This is a research/validation tool, not a trading system, execution
  engine, or source of investment advice. It makes no network calls and
  does not place orders.
