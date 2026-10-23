"""Deterministic demo: print a short edge-decay summary for the gallery.

Research-only: synthetic data, no network access, no trading.
"""
from edgedecay.cli import DEFAULT_MIN_T, format_summary, run_gallery


def main() -> None:
    results = run_gallery()
    for name, (scenario, curve, hl, be, t0) in results.items():
        hl_str, be_str = format_summary(hl, be, t0, DEFAULT_MIN_T)
        s0 = curve["sharpe"].iloc[0]
        s_last = curve["sharpe"].iloc[-1]
        print(f"[{name}] {scenario.description}")
        print(
            f"  sharpe(delay=0)={s0:.3f}  sharpe(delay={curve.index[-1]})={s_last:.3f}  "
            f"half_life={hl_str}  breakeven_cost={be_str}"
        )


if __name__ == "__main__":
    main()
