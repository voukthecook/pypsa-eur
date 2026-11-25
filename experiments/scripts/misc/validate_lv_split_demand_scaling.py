"""
Validate the LV-bus split (steps 1 and 2 of the validation plan).

Usage
-----
# Step 1: structural checks on the prepared (unsolved) network
python validate_lv_split.py prepared.nc --solar 1.2 0.8 --demand 1.2 0.8 \
    [--baseline prepared_unsplit.nc]

# Step 2: symmetric equivalence of two SOLVED networks (split vs unsplit)
python validate_lv_split.py --compare solved_split.nc solved_unsplit.nc [--tol 1e-4]

Exit code is 1 if any check fails.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import pypsa

SUFFIXES = (" a", " b")


def _normalise(d):
    s = sum(d)
    return tuple(2 * x / s for x in d)


def _on_bus(n, bus):
    """Index of elements attached to `bus`, per component."""
    return {
        "Load": n.loads.index[n.loads.bus == bus],
        "Generator": n.generators.index[n.generators.bus == bus],
        "Link": n.links.index[(n.links.bus0 == bus) | (n.links.bus1 == bus)],
        "Store": n.stores.index[n.stores.bus == bus],
    }


def _load_energy(n, loads):
    loads = pd.Index(loads)
    in_ts = loads.isin(n.loads_t.p_set.columns)
    ts = n.loads_t.p_set[loads[in_ts]].sum().sum()
    static = n.loads.loc[loads[~in_ts], "p_set"].sum() * len(n.snapshots)
    return float(ts + static)


def _elec_loads(n, loads):
    return [l for l in loads if "electric" in str(n.loads.at[l, "carrier"])]


def _referenced_buses(n):
    ref = set()
    for comp, cols in [("links", ["bus0", "bus1"]), ("loads", ["bus"]),
                       ("generators", ["bus"]), ("stores", ["bus"]),
                       ("storage_units", ["bus"]), ("lines", ["bus0", "bus1"]),
                       ("transformers", ["bus0", "bus1"])]:
        df = getattr(n, comp)
        for c in cols:
            if c in df:
                ref |= set(df[c])
    return ref


def check_structure(n, solar, demand, baseline=None, tol=1e-6):
    results = []

    def add(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    demand = _normalise(demand)

    # 1. consistency check
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n.consistency_check()
        add("pypsa consistency_check", True, f"{len(w)} warning(s)" if w else "")
    except Exception as e:
        add("pypsa consistency_check", False, str(e)[:200])

    # 2. no leftover unsplit LV / home battery buses
    lv = n.buses.index[n.buses.carrier == "low voltage"]
    bad = [b for b in lv if not b.endswith(SUFFIXES)]
    add("no unsplit 'low voltage' buses left", not bad, str(bad[:5]))
    batt = [b for b in n.buses.index if "home battery" in b]
    bad_b = [b for b in batt if not b.endswith(SUFFIXES)]
    add("no unsplit 'home battery' buses left", not bad_b, str(bad_b[:5]))

    # 3. no orphan buses
    orphans = set(n.buses.index) - _referenced_buses(n)
    nb = pypsa.Network(baseline) if baseline is not None else None
    pre_existing = set()
    if nb is not None:
        pre_existing = set(nb.buses.index) - _referenced_buses(nb)
    new_orphans = sorted(orphans - pre_existing)
    add("no new orphan buses (vs baseline)" if nb is not None else "no orphan buses",
        not new_orphans, str(new_orphans[:5]))
    if pre_existing & orphans:
        add("orphan buses also present in baseline (not caused by split)", True,
            str(sorted(pre_existing & orphans)[:5]))

    # 4. per node checks
    nodes = sorted({b[: -len(" low voltage a")] for b in lv if b.endswith(" low voltage a")})
    add("at least one split node found", len(nodes) > 0, f"{len(nodes)} nodes")
    bad_pair, bad_count, bad_dem, bad_sol, bad_prof = [], [], [], [], []
    for node in nodes:
        ba, bb = f"{node} low voltage a", f"{node} low voltage b"
        if bb not in n.buses.index:
            bad_pair.append(node)
            continue
        ea, eb = _on_bus(n, ba), _on_bus(n, bb)

        # same number of elements per carrier on each half
        for comp in ea:
            df = getattr(n, comp.lower() + "s")
            ca = df.loc[ea[comp], "carrier"].value_counts().sort_index()
            cb = df.loc[eb[comp], "carrier"].value_counts().sort_index()
            if not ca.equals(cb):
                bad_count.append((node, comp))

        # demand split follows demand_scaling
        la, lb = _elec_loads(n, ea["Load"]), _elec_loads(n, eb["Load"])
        Ea, Eb = _load_energy(n, la), _load_energy(n, lb)
        if Ea + Eb > 0 and abs(Ea / (Ea + Eb) - demand[0] / 2) > 1e-6:
            bad_dem.append((node, round(Ea / (Ea + Eb), 4)))

        # solar potential split + identical availability profiles
        sa = [g for g in ea["Generator"] if n.generators.at[g, "carrier"] == "solar rooftop"]
        sb = [g for g in eb["Generator"] if n.generators.at[g, "carrier"] == "solar rooftop"]
        if sa and sb:
            pa = n.generators.loc[sa, "p_nom_max"].sum()
            pb = n.generators.loc[sb, "p_nom_max"].sum()
            if np.isfinite(pa) and np.isfinite(pb) and pb > 0 and \
                    abs(pa / pb - solar[0] / solar[1]) > 1e-6 * max(1, solar[0] / solar[1]):
                bad_sol.append((node, round(pa / pb, 4)))
            for g1, g2 in zip(sorted(sa), sorted(sb)):
                if g1 in n.generators_t.p_max_pu and g2 in n.generators_t.p_max_pu:
                    if not np.allclose(n.generators_t.p_max_pu[g1], n.generators_t.p_max_pu[g2]):
                        bad_prof.append(node)

    add("both halves (a, b) exist for every node", not bad_pair, str(bad_pair[:5]))
    add("same elements per carrier on both halves", not bad_count, str(bad_count[:5]))
    add(f"electricity demand share of 'a' == {demand[0] / 2:.3f}", not bad_dem, str(bad_dem[:5]))
    add(f"rooftop solar p_nom_max ratio a/b == {solar[0] / solar[1]:.3f}", not bad_sol, str(bad_sol[:5]))
    add("rooftop solar p_max_pu identical on a and b", not bad_prof, str(sorted(set(bad_prof))[:5]))

    # 5. against an unsplit baseline: node totals conserved
    if baseline is not None:
        lvb = nb.buses.index[nb.buses.carrier == "low voltage"]
        bad_tot = []
        for node in nodes:
            b0 = f"{node} low voltage"
            if b0 not in lvb:
                continue
            e0 = _on_bus(nb, b0)
            E0 = _load_energy(nb, _elec_loads(nb, e0["Load"]))
            E1 = sum(
                _load_energy(n, _elec_loads(n, _on_bus(n, f"{node} low voltage {s}")["Load"]))
                for s in "ab"
            )
            if abs(E0 - E1) > tol * max(abs(E0), 1):
                bad_tot.append((node, E0, E1))
        add("node electricity demand conserved vs baseline", not bad_tot, str(bad_tot[:3]))
    return results


def compare_solved(path_a, path_b, tol):
    a, b = pypsa.Network(path_a), pypsa.Network(path_b)
    out = []
    oa, ob = float(a.objective), float(b.objective)
    rel = abs(oa - ob) / max(abs(ob), 1e-9)
    out.append(("objective equal", rel < tol, f"{oa:.6e} vs {ob:.6e} (rel diff {rel:.2e})"))
    for comp, col in [("generators", "p_nom_opt"), ("links", "p_nom_opt"), ("stores", "e_nom_opt")]:
        sa = getattr(a, comp).groupby("carrier")[col].sum()
        sb = getattr(b, comp).groupby("carrier")[col].sum()
        df = pd.concat([sa, sb], axis=1, keys=["split", "unsplit"]).fillna(0)
        df["rel"] = (df.split - df.unsplit).abs() / df.unsplit.abs().clip(lower=1)
        bad = df[df.rel > max(tol, 1e-3)]
        out.append((f"{comp} {col} by carrier equal", bad.empty,
                    bad.round(3).to_string() if not bad.empty else ""))
    return out


def report(results):
    ok_all = True
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
        ok_all &= ok
    return ok_all


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("network", nargs="?")
    p.add_argument("--solar", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--demand", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--baseline")
    p.add_argument("--compare", nargs=2, metavar=("SPLIT_SOLVED", "UNSPLIT_SOLVED"))
    p.add_argument("--tol", type=float, default=1e-4)
    a = p.parse_args()
    if a.compare:
        ok = report(compare_solved(*a.compare, a.tol))
    else:
        n = pypsa.Network(a.network)
        ok = report(check_structure(n, tuple(a.solar), tuple(a.demand), a.baseline))
    sys.exit(0 if ok else 1)