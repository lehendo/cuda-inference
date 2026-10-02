"""Smoke test of aggregation + plotting on the SYNTHETIC fixture only (run: python -m pytest analysis/test_analysis.py)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def test_pipeline_on_synthetic_fixture(tmp_path):
    run("fixtures/make_synthetic_fixture.py")
    raw = os.path.join(HERE, "fixtures", "out", "raw")
    out = tmp_path / "processed"
    figs = tmp_path / "figures"
    run("aggregate_results.py", "--raw", raw, "--out", str(out))
    run("make_figures.py", "--raw", raw, "--figures", str(figs))
    assert (out / "aggregate_speedups.csv").exists()
    assert any(f.startswith("fig09_roofline") for f in os.listdir(figs))
    assert any(f.startswith("fig10_winners") for f in os.listdir(figs))
