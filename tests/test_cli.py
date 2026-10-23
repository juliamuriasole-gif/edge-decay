import pandas as pd
import pytest

from edgedecay.cli import DEFAULT_MIN_T, format_summary, main, run_gallery
from edgedecay.core import backtest, breakeven_cost_bps


def test_cli_writes_report_and_plot(tmp_path):
    out_dir = tmp_path / "out"
    main(["--output-dir", str(out_dir)])

    assert (out_dir / "report.md").exists()
    assert (out_dir / "decay.png").exists()
    report = (out_dir / "report.md").read_text()
    assert "## noise" in report
    assert "no significant edge" in report  # the noise control should be gated


def test_cli_exits_cleanly_on_unwritable_output_dir(tmp_path, capsys):
    # `blocker` is a regular file, so os.makedirs(blocker / "report") must fail
    # with an OSError (NotADirectoryError), independent of filesystem permissions.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    bad_output_dir = blocker / "report"

    with pytest.raises(SystemExit) as exc_info:
        main(["--output-dir", str(bad_output_dir)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "edgedecay" in captured.err


def test_format_summary_gates_on_significance():
    hl_str, be_str = format_summary(hl=5.0, be=12.3, t0=0.5, min_t=2.0)
    assert "n/a" in hl_str
    assert "no significant edge" in hl_str
    assert hl_str == be_str


def test_format_summary_passes_through_when_significant():
    hl_str, be_str = format_summary(hl=5.0, be=12.3, t0=3.0, min_t=2.0)
    assert hl_str == "5.00 bars"
    assert be_str == "12.30 bps"


def test_format_summary_reports_na_for_no_breakeven_edge():
    hl_str, be_str = format_summary(hl=None, be=None, t0=3.0, min_t=2.0)
    assert hl_str == "none within grid"
    assert be_str == "n/a (no edge to spend on costs)"


def test_noise_scenario_is_not_significant_and_gated():
    results = run_gallery()
    _, _curve, hl, be, t0 = results["noise"]
    assert t0 < DEFAULT_MIN_T
    hl_str, be_str = format_summary(hl, be, t0, DEFAULT_MIN_T)
    assert "no significant edge" in hl_str
    assert "no significant edge" in be_str


def test_breakeven_cost_is_none_when_gross_mean_not_positive():
    prices = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0])  # steadily losing

    def always_long(p):
        return pd.Series(1.0, index=p.index)

    be = breakeven_cost_bps(prices, always_long, delay=0)
    assert be is None


def test_backtest_rejects_negative_delay():
    prices = pd.Series([100.0, 101.0, 102.0])
    positions = pd.Series([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        backtest(prices, positions, delay=-1)


def test_backtest_rejects_non_integer_delay():
    prices = pd.Series([100.0, 101.0, 102.0])
    positions = pd.Series([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        backtest(prices, positions, delay=1.5)
