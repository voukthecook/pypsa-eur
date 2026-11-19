import pandas as pd
import pytest

from scripts.lv_split.compare_runs import main
from scripts.lv_profile_split import apply_lv_profile_split
from test.test_lv_profile_shapes import _config, _network, _shape_table

TABLE = "data/lv_profile_shapes/lpg_electricity.csv"


def _pair(tmp_path):
    control = apply_lv_profile_split(_network(), _config())
    shaped = apply_lv_profile_split(
        _network(), _config(shapes={"electricity": _shape_table(0.8)})
    )
    c, s = tmp_path / "control.nc", tmp_path / "shaped.nc"
    control.export_to_netcdf(c)
    shaped.export_to_netcdf(s)
    return c, s, shaped


def test_command_passes_on_correct_shapes(tmp_path, capsys) -> None:
    c, s, _ = _pair(tmp_path)
    code = main([str(c), str(s), "--out", str(tmp_path / "out")])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "[PASS] Household node total unchanged" in text
    assert "[PASS] Profiles follow different shapes" in text
    assert (tmp_path / "out" / "profile_summary.csv").exists()


def test_command_fails_when_demand_is_altered(tmp_path, capsys) -> None:
    c, s, shaped = _pair(tmp_path)
    shaped.loads_t.p_set["electric demand urban"] *= 1.2
    shaped.export_to_netcdf(s)
    code = main([str(c), str(s), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "[FAIL] Household node total unchanged" in capsys.readouterr().out


def test_command_fails_when_shapes_missing(tmp_path, capsys) -> None:
    c, _, _ = _pair(tmp_path)
    code = main([str(c), str(c), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "[FAIL] Profiles follow different shapes" in capsys.readouterr().out


def test_command_checks_ratios_against_table(tmp_path, capsys) -> None:
    control = apply_lv_profile_split(_network(), _config())
    shaped = apply_lv_profile_split(
        _network(), _config(shapes={"electricity": TABLE})
    )
    c, s = tmp_path / "c.nc", tmp_path / "s.nc"
    control.export_to_netcdf(c)
    shaped.export_to_netcdf(s)
    code = main([str(c), str(s), "--table", TABLE, "--out", str(tmp_path / "o")])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "[PASS] Profile-to-profile ratios follow the shape table" in text


def test_diagnostics_when_nothing_is_split(tmp_path) -> None:
    n = _network()
    f = tmp_path / "plain.nc"
    n.export_to_netcdf(f)
    with pytest.raises(ValueError, match="nothing is split"):
        main([str(f), str(f), "--out", str(tmp_path / "o")])


def test_diagnostics_suggest_carrier(tmp_path) -> None:
    c, s, _ = _pair(tmp_path)
    with pytest.raises(ValueError, match="rerun with --carriers"):
        main([str(c), str(s), "--carriers", "nonexistent", "--out", str(tmp_path / "o")])


def test_unsplit_control_is_compared_after_adding_profiles_up(tmp_path, capsys) -> None:
    control = _network()  # baseline run without any split
    shaped = apply_lv_profile_split(
        _network(), _config(shapes={"electricity": TABLE})
    )
    c, s = tmp_path / "c.nc", tmp_path / "s.nc"
    control.export_to_netcdf(c)
    shaped.export_to_netcdf(s)
    code = main([str(c), str(s), "--table", TABLE, "--out", str(tmp_path / "o")])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "[PASS] Household node total unchanged" in text
    assert "[PASS] Other loads (EV, heat, ...) unchanged" in text
    assert "[PASS] Profile-to-profile ratios follow the shape table" in text
    assert "YEARLY SHARE" in text


def test_unsplit_control_detects_changed_node_total(tmp_path, capsys) -> None:
    control = _network()
    shaped = apply_lv_profile_split(_network(), _config(shapes={"electricity": TABLE}))
    shaped.loads_t.p_set["electric demand rural"] *= 1.1
    c, s = tmp_path / "c.nc", tmp_path / "s.nc"
    control.export_to_netcdf(c)
    shaped.export_to_netcdf(s)
    assert main([str(c), str(s), "--out", str(tmp_path / "o")]) == 1
    assert "[FAIL] Household node total unchanged" in capsys.readouterr().out
