"""Run the edge-decay demo gallery (fast / slow / noise) and write a report.

Research-only: synthetic data, no network access, no trading.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import breakeven_cost_bps, decay_curve, half_life
from .synthetic import fast_alpha_signal, generate_prices, slow_trend_signal

DELAYS = range(0, 31)
DEFAULT_MIN_T = 2.0


@dataclass
class Scenario:
    name: str
    description: str
    prices_fn: callable
    signal_fn: callable


GALLERY = [
    Scenario(
        name="fast",
        description="fast-decaying hidden alpha (phi=0.15) + 1-bar-return signal",
        prices_fn=partial(generate_prices, n=6000, phi=0.15, alpha_sigma=0.008, noise_sigma=0.01, seed=1),
        signal_fn=partial(fast_alpha_signal, k=15.0),
    ),
    Scenario(
        name="slow",
        description="persistent hidden alpha (phi=0.95) + 20-bar trend signal",
        prices_fn=partial(generate_prices, n=6000, phi=0.95, alpha_sigma=0.0012, noise_sigma=0.01, seed=2),
        signal_fn=partial(slow_trend_signal, window=20, k=20.0),
    ),
    Scenario(
        name="noise",
        description="pure noise (no hidden alpha) + the same 20-bar trend signal, as a control",
        prices_fn=partial(generate_prices, n=6000, phi=0.9, alpha_sigma=0.0, noise_sigma=0.01, seed=3),
        signal_fn=partial(slow_trend_signal, window=20, k=20.0),
    ),
]


def run_gallery(periods_per_year: int = 252):
    """Run the gallery and return {name: (scenario, curve, half_life, breakeven_bps, t_stat)}.

    `t_stat` is the delay-0 Sharpe t-stat (see `decay_curve`), used to gate
    whether half-life/breakeven are meaningful to report (see `format_summary`).
    """
    results = {}
    for scenario in GALLERY:
        prices = scenario.prices_fn()
        curve = decay_curve(
            prices, scenario.signal_fn, delays=DELAYS, cost_bps=0.0, periods_per_year=periods_per_year
        )
        hl = half_life(curve)
        be = breakeven_cost_bps(prices, scenario.signal_fn, delay=0)
        t0 = float(curve["t_stat"].iloc[0])
        results[scenario.name] = (scenario, curve, hl, be, t0)
    return results


def format_summary(hl, be, t0: float, min_t: float = DEFAULT_MIN_T):
    """Format (half_life, breakeven) strings, gated on delay-0 significance.

    Below `min_t`, there is no statistically significant edge at delay 0, so
    reporting a half-life or breakeven cost for it would be reporting a
    noise artifact; both are reported as "n/a" instead. The raw decay curve
    itself is unaffected and should still be shown alongside this.
    """
    if not (t0 >= min_t):
        msg = f"n/a (no significant edge, t={t0:.2f})"
        return msg, msg

    hl_str = "none within grid" if hl is None else f"{hl:.2f} bars"
    if be is None:
        be_str = "n/a (no edge to spend on costs)"
    elif be == float("inf"):
        be_str = "inf (no turnover)"
    else:
        be_str = f"{be:.2f} bps"
    return hl_str, be_str


def write_report(results, report_dir: str, min_t: float = DEFAULT_MIN_T) -> None:
    os.makedirs(report_dir, exist_ok=True)

    lines = ["# Edge Decay Report", ""]
    lines.append(
        "Research-only demo: synthetic price series with a seeded, injected "
        "hidden-alpha component, measuring how each strategy's Sharpe ratio "
        "decays as execution is delayed by k bars. Half-life and breakeven "
        f"cost are only reported when the delay-0 Sharpe t-stat is >= {min_t:.1f}; "
        "below that threshold there is no statistically significant edge to "
        "measure decay or cost tolerance for, so both are shown as n/a (the "
        "raw decay curve below is unaffected)."
    )
    lines.append("")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, (scenario, curve, hl, be, t0) in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(scenario.description)
        lines.append("")
        hl_str, be_str = format_summary(hl, be, t0, min_t)
        lines.append(f"- delay-0 t-stat: {t0:.2f}")
        lines.append(f"- half-life: {hl_str}")
        lines.append(f"- breakeven cost (at delay 0): {be_str}")
        lines.append("")
        lines.append("| delay | sharpe | total_return | turnover |")
        lines.append("|---:|---:|---:|---:|")
        step = max(1, len(curve) // 15)
        for delay, row in curve.iloc[::step].iterrows():
            lines.append(
                f"| {delay} | {row['sharpe']:.3f} | {row['total_return']:.3f} | {row['turnover']:.4f} |"
            )
        lines.append("")

        ax.plot(curve.index, curve["sharpe"], marker="o", markersize=2, label=name)

    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("execution delay (bars)")
    ax.set_ylabel("annualised Sharpe")
    ax.set_title("Sharpe vs. execution delay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "decay.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(report_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="edgedecay",
        description="Run the edge-decay demo gallery and write a report.",
    )
    parser.add_argument(
        "--output-dir",
        default="report",
        help="directory to write report.md and decay.png into (default: report)",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="bars per year, used to annualize Sharpe (default: 252, i.e. daily bars)",
    )
    parser.add_argument(
        "--min-t",
        type=float,
        default=DEFAULT_MIN_T,
        help="minimum delay-0 Sharpe t-stat required to report half-life/breakeven (default: 2.0)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    results = run_gallery(periods_per_year=args.periods_per_year)

    try:
        write_report(results, args.output_dir, min_t=args.min_t)
    except OSError as exc:
        print(f"edgedecay: could not write report to {args.output_dir!r}: {exc.strerror or exc}", file=sys.stderr)
        sys.exit(2)

    for name, (scenario, curve, hl, be, t0) in results.items():
        hl_str, be_str = format_summary(hl, be, t0, args.min_t)
        print(f"{name}: sharpe(delay=0)={curve['sharpe'].iloc[0]:.3f} half_life={hl_str} breakeven={be_str}")
    print(f"wrote {args.output_dir}/report.md and {args.output_dir}/decay.png")


if __name__ == "__main__":
    main()
