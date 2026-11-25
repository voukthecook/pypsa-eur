import pandas as pd

from scripts.lv_profile_split import apply_lv_profile_split
from experiments.scripts.lv_split.validate_shapes import main, validate
from test.test_compare_runs import TABLE, _local_shapes
from test.test_lv_profile_shapes import _config, _network


def _pair(tmp_path, shaped_cfg):
    control = apply_lv_profile_split(_network(), _config())
    shaped = apply_lv_profile_split(_network(), shaped_cfg)
    c, s = tmp_path / "c.nc", tmp_path / "s.nc"
    control.export_to_netcdf(c)
    shaped.export_to_netcdf(s)
    return c, s, shaped


def test_correct_local_time_shapes_pass_every_check(tmp_path, capsys) -> None:
    c, s, _ = _pair(tmp_path, _config(resolved_shapes=_local_shapes()))
    code = main([str(c), str(s), "--table", TABLE, "--out", str(tmp_path / "o")])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "A. Conservation" in text and "C. Local clock" in text
    assert (tmp_path / "o" / "shape_fidelity.csv").exists()


def test_altered_demand_fails_conservation(tmp_path, capsys) -> None:
    c, s, shaped = _pair(tmp_path, _config(resolved_shapes=_local_shapes()))
    shaped.loads_t.p_set["electric demand urban"] *= 1.2
    shaped.export_to_netcdf(s)
    assert main([str(c), str(s), "--table", TABLE, "--out", str(tmp_path / "o")]) == 1
    assert "[FAIL] Household node total unchanged" in capsys.readouterr().out


def test_wrong_clock_is_detected(tmp_path) -> None:
    # network built with the UTC lookup, but validated as if the country were in Tokyo
    control = apply_lv_profile_split(_network(), _config())
    shaped = apply_lv_profile_split(_network(), _config(shapes={"electricity": TABLE}))
    result = validate(
        control, shaped, pd.read_csv(TABLE), timezones={"NO": "Asia/Tokyo"}
    )
    failed = [n for g, n, ok, _ in result["checks"] if not ok]
    assert any("Local-time reading" in n for n in failed)


def test_unsplit_control_works(tmp_path) -> None:
    shaped = apply_lv_profile_split(_network(), _config(resolved_shapes=_local_shapes()))
    result = validate(_network(), shaped, pd.read_csv(TABLE))
    assert all(ok for _, _, ok, _ in result["checks"]), result["checks"]


def test_implied_household_fraction_is_recovered() -> None:
    for fraction in (1.0, 0.5):
        shaped = apply_lv_profile_split(
            _network(),
            _config(resolved_shapes=_local_shapes(), household_fraction=fraction),
        )
        result = validate(_network(), shaped, pd.read_csv(TABLE))
        assert abs(result["implied_fraction"]["NO"] - fraction) < 0.15, result["implied_fraction"]
