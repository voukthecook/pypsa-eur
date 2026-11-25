import numpy as np

from scripts.lv_profile_split import apply_lv_profile_split
from experiments.scripts.lv_split.plot_demand import analyse, main
from test.test_compare_runs import _local_shapes
from test.test_lv_profile_shapes import SHARES, _config, _network


def _with_charging(n):
    for link in n.links.index[n.links.carrier == "BEV charger"]:
        share = SHARES.get(link.rsplit(" ", 1)[-1], 1.0)
        n.links_t.p0[link] = share * np.linspace(1.0, 3.0, len(n.snapshots))
    return n


def test_analyse_unsplit_control_and_shaped(tmp_path) -> None:
    control = _with_charging(_network())
    shaped = _with_charging(
        apply_lv_profile_split(_network(), _config(resolved_shapes=_local_shapes()))
    )
    r = analyse(control, shaped, ["electricity"])
    # household electricity: unshaped profiles add up to the control, shaped too
    np.testing.assert_allclose(
        r["hh_c"].sum(axis=1), r["hh_s"].sum(axis=1), rtol=1e-9
    )
    assert r["effect"]["household_non_coincidence_pct"].iloc[0] > 0
    # EV charging is in the LV total, so the whole-level effect is diluted
    assert "EV charging" in r["summary"].index.str.replace("energy_share_", "")
    assert (
        r["effect"]["LV_total_non_coincidence_pct"].iloc[0]
        < r["effect"]["household_non_coincidence_pct"].iloc[0] + 1e-9
    )


def test_cli_writes_figures(tmp_path) -> None:
    control = _with_charging(_network())
    shaped = _with_charging(
        apply_lv_profile_split(_network(), _config(resolved_shapes=_local_shapes()))
    )
    c, s = tmp_path / "c.nc", tmp_path / "s.nc"
    control.export_to_netcdf(c)
    shaped.export_to_netcdf(s)
    assert main([str(c), str(s), "--out", str(tmp_path / "o")]) == 0
    for name in ("household_profiles.png", "lv_level.png", "household_weeks.png"):
        assert (tmp_path / "o" / name).exists()
