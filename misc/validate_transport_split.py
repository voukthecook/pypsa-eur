"""
Validate a solved transport (EV) split network against a reference run.

Usage:
    python validate_transport_split.py REF.nc SPLIT.nc [--config cfg.yaml] [--rtol 1e-3]

Reference = same scenario with the electricity (LV) split ON, the LV constraint set
exactly as in the split run, and ONLY the EV split disabled (shared EV battery bus,
EV load and EV store per node; chargers/V2G already on the LV halves).
Split     = same scenario with the EV split enabled (split_transport).

With a neutral config (all EV scalings (1, 1)) the split network must reproduce the
reference: same objective, same capacity and energy totals, identical prices on
half a and half b. With a non-neutral config only the structural, conservation and
share checks apply.

Config keys (nested dict or flat keys under `sector:`; same style as the heat split):
    electricity_distribution_grid_split_lv_ev:
        ev_scaling / ev_charger_scaling / ev_v2g_scaling / ev_store_scaling
    electricity_distribution_grid_split_lv_ev_scaling  (flat form, etc.)
"""

import argparse
import sys

import numpy as np
import pypsa
import yaml

# Self-contained: same carrier names as split_transport in prepare_sector_network.py
HALVES = ("a", "b")
EV_LOAD = "land transport EV"
EV_STORE = "EV battery"
CHARGER, V2G = "BEV charger", "V2G"
LV_CARRIER = "low voltage"
DIST_CARRIER = "electricity distribution grid"
SCALING_KEYS = ("ev_scaling", "ev_charger_scaling", "ev_v2g_scaling", "ev_store_scaling")

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


# ---------------------------------------------------------------- helpers
def _norm(v):
    v = (1.0, 1.0) if v is None else tuple(float(x) for x in v)
    t = sum(v)
    return tuple(2 * x / t for x in v) if t > 0 else v


def resolve(tcfg):
    """Per-component (a, b) scalings; None follows ev_scaling (as split_transport does)."""
    ev = _norm(tcfg.get("ev_scaling"))
    pick = lambda k: _norm(tcfg[k]) if tcfg.get(k) else ev
    return {"load": ev, "charger": pick("ev_charger_scaling"),
            "v2g": pick("ev_v2g_scaling"), "store": pick("ev_store_scaling")}


def is_neutral(sc):
    return all(np.allclose(v, 1.0) for v in sc.values())


def half_of(name):
    return "a" if str(name).endswith(" a") else "b" if str(name).endswith(" b") else None


def chain_buses(n):
    return n.buses.index[n.buses.carrier == EV_STORE]


def lv_buses(n):
    return n.buses.index[n.buses.carrier == LV_CARRIER]


def load_energy(n, ids):
    w = n.snapshot_weightings.generators
    tot = 0.0
    for i in ids:
        if i in n.loads_t.p_set.columns:
            tot += float(n.loads_t.p_set[i].mul(w).sum())
        else:
            tot += float(n.loads.at[i, "p_set"] * w.sum())
    return tot


def transport_demand(n):
    """Energy per land-transport load carrier (EV and non-EV)."""
    out = {}
    for c in sorted({c for c in n.loads.carrier.astype(str) if c.startswith("land transport")}):
        out[c] = load_energy(n, n.loads.index[n.loads.carrier == c])
    return out


def demand_by_half(n):
    """EV demand energy on half a and half b (split networks only)."""
    out = {"a": 0.0, "b": 0.0}
    for i in n.loads.index[n.loads.carrier == EV_LOAD]:
        h = half_of(n.loads.at[i, "bus"])
        if h:
            out[h] += load_energy(n, [i])
    return out


def ev_port(carrier):
    """Column holding the EV-side bus of a charger / V2G link (and the LV-side one)."""
    return ("bus1", "bus0") if carrier == CHARGER else ("bus0", "bus1")


def capacity_by_half(n):
    """Installed (static) EV capacity per half: store e_nom, charger p_nom, V2G p_nom."""
    out = {k: {"a": 0.0, "b": 0.0} for k in ("store", "charger", "v2g")}
    st = n.stores[n.stores.carrier == EV_STORE]
    for i in st.index:
        h = half_of(st.at[i, "bus"])
        if h:
            out["store"][h] += float(st.at[i, "e_nom"])
    for carrier, key in ((CHARGER, "charger"), (V2G, "v2g")):
        ls = n.links[n.links.carrier == carrier]
        ev_col, _ = ev_port(carrier)
        for i in ls.index:
            h = half_of(ls.at[i, ev_col])
            if h:
                out[key][h] += float(ls.at[i, "p_nom"])
    return out


def installed(n):
    return {CHARGER: float(n.links.loc[n.links.carrier == CHARGER, "p_nom"].sum()),
            V2G: float(n.links.loc[n.links.carrier == V2G, "p_nom"].sum()),
            EV_STORE: float(n.stores.loc[n.stores.carrier == EV_STORE, "e_nom"].sum())}


def capacities(n):
    """Optimal capacities of the EV chain and the LV distribution links."""
    out = {}
    for name, df, nom in ((CHARGER, n.links[n.links.carrier == CHARGER], "p_nom"),
                          (V2G, n.links[n.links.carrier == V2G], "p_nom"),
                          (EV_STORE, n.stores[n.stores.carrier == EV_STORE], "e_nom"),
                          ("distribution grid", n.links[n.links.carrier == DIST_CARRIER], "p_nom")):
        col = nom + "_opt" if nom + "_opt" in df.columns else nom
        out[name] = float(df[col].sum())
    return out


def link_energy(n, carrier, by_half=False):
    w = n.snapshot_weightings.generators
    ev_col, _ = ev_port(carrier)
    out = {"a": 0.0, "b": 0.0, "other": 0.0}
    for l in n.links.index[n.links.carrier == carrier]:
        if l in n.links_t.p0.columns:
            out[half_of(n.links.at[l, ev_col]) or "other"] += float(n.links_t.p0[l].mul(w).sum())
    return out if by_half else sum(out.values())


def prices(n, buses):
    return n.buses_t.marginal_price[[b for b in buses if b in n.buses_t.marginal_price]]


def lv_constrained(n):
    d = n.links.loc[n.links.carrier == DIST_CARRIER, "p_nom_extendable"].astype(bool)
    return bool((~d).any())


def dist_sig(n):
    d = n.links[n.links.carrier == DIST_CARRIER]
    return sorted({(bool(e), round(float(c), 3)) for e, c in zip(d.p_nom_extendable, d.capital_cost)})


def carrier_table(n):
    """Optimal capacity, capex and dispatch energy per component|carrier (for diagnosis)."""
    w = n.snapshot_weightings.generators
    out = {}
    for comp, plural, nom, dispatch in (("Generator", "generators", "p_nom", "p"),
                                        ("Link", "links", "p_nom", "p0"),
                                        ("Store", "stores", "e_nom", None)):
        df = getattr(n, plural)
        col = nom + "_opt" if nom + "_opt" in df.columns else nom
        ext = df[nom + "_extendable"].astype(bool)
        capex = (df.capital_cost.where(ext, 0.0) * df[col]).groupby(df.carrier).sum()
        cap = df[col].groupby(df.carrier).sum()
        energy = {}
        if dispatch:
            t = getattr(n, plural + "_t")[dispatch]
            cols = [c for c in t.columns if c in df.index]
            energy = t[cols].mul(w, axis=0).sum().groupby(df.carrier.reindex(cols)).sum()
        for c in cap.index:
            out[f"{comp}|{c}"] = (float(cap[c]), float(capex.get(c, 0.0)),
                                  float(energy[c]) if c in energy else 0.0)
    return out


def diagnose(ref, spl, rtol, top=8):
    """Localise WHERE a neutral split differs from the reference (carrier level)."""
    tr, ts = carrier_table(ref), carrier_table(spl)
    rows = []
    for k in set(tr) | set(ts):
        (cr, xr, er), (cs, xs, es) = tr.get(k, (0, 0, 0)), ts.get(k, (0, 0, 0))
        rows.append((abs(xs - xr), k, cr, cs, xr, xs, er, es))
    rows.sort(reverse=True)
    out = [("Capex ref -> split (total)", INFO,
            f"{sum(v[1] for v in tr.values()):.6g} -> {sum(v[1] for v in ts.values()):.6g}")]
    for _, k, cr, cs, xr, xs, er, es in rows[:top]:
        if (abs(xs - xr) <= rtol * max(1.0, abs(xr))
                and np.isclose(er, es, rtol=rtol, atol=rtol * max(1.0, abs(er)))
                and np.isclose(cr, cs, rtol=rtol, atol=1e-6)):
            continue
        out.append((f"Diff {k}", INFO,
                    f"cap {cr:.5g} -> {cs:.5g} | capex {xr:.5g} -> {xs:.5g} | energy {er:.5g} -> {es:.5g}"))
    return out


# ---------------------------------------------------------------- validation
def validate(ref, spl, sc, rtol=1e-3):
    res = []

    def add(name, ok, detail=""):
        res.append((name, PASS if ok else FAIL, detail))

    neutral = is_neutral(sc)
    res.append(("Config neutral (all EV scalings 1,1)", INFO, str(neutral)))
    res.append(("Resolved scalings", INFO, str(sc)))

    # ---- is there an EV chain to compare at all? (guards against empty-not-passed)
    for label, net in (("reference", ref), ("split", spl)):
        has = {"EV load": (net.loads.carrier == EV_LOAD).any(),
               "EV store": (net.stores.carrier == EV_STORE).any(),
               CHARGER: (net.links.carrier == CHARGER).any()}
        ok = all(has.values())
        add(f"EV chain recognised in {label}", ok,
            "" if ok else f"missing {[k for k, v in has.items() if not v]} -> checks below would be "
            "EMPTY, not passed; carrier names differ or BEV is disabled in this scenario")
    if not (ref.loads.carrier == EV_LOAD).any():
        return res
    has_v2g = (ref.links.carrier == V2G).any()
    if not has_v2g:
        res.append(("V2G", INFO, "no V2G links in reference (v2g disabled?) -> V2G checks trivial"))

    # ---- reference must be the right kind of reference (stop early otherwise)
    ref_split = [b for b in chain_buses(ref) if half_of(b)]
    add("Reference is unsplit (EV buses have no a/b suffix)", not ref_split,
        "" if not ref_split else f"{len(ref_split)} EV buses already end in a/b, e.g. "
        f"{ref_split[:2]} -> the reference was run WITH the EV split; re-run with it disabled")
    lv_ref = lv_buses(ref)
    lv_ok = len(lv_ref) > 0 and all(half_of(b) for b in lv_ref)
    add("Reference LV buses are split (LV split on, EV split off)", lv_ok,
        f"{len(lv_ref)} LV buses, e.g. {list(lv_ref[:2])}; the reference needs the electricity "
        "(LV) split ON and only the EV split disabled")
    if ref_split or not lv_ok:
        return res

    # ---- structural
    ports = [c for c in spl.links.columns if c.startswith("bus") and c[3:].isdigit()]
    orphans = [(p, b) for p in ports for b in spl.links[p]
               if b not in ("", None) and b not in spl.buses.index]
    for comp in ("loads", "generators", "stores"):
        orphans += [(comp, b) for b in getattr(spl, comp).bus if b not in spl.buses.index]
    add("No component attached to a missing bus", not orphans, str(orphans[:3]))

    left = [b for b in chain_buses(ref) if b in spl.buses.index]
    add("Original EV buses removed", not left, str(left[:3]))

    new_ev = chain_buses(spl)
    add("EV buses are all on halves a/b",
        all(half_of(b) for b in new_ev) and len(new_ev) == 2 * len(chain_buses(ref)),
        f"{len(new_ev)} EV buses vs {len(chain_buses(ref))} in reference")
    lvl = lv_buses(spl)
    add("LV buses are all on halves a/b", len(lvl) > 0 and all(half_of(b) for b in lvl),
        f"{len(lvl)} LV buses")

    for comp in ("loads", "stores", "links"):
        dup = getattr(spl, comp).index[getattr(spl, comp).index.duplicated()]
        add(f"No duplicated {comp} names", len(dup) == 0)

    # component identity, mirroring split_transport:
    #   loads/stores are cloned      <name> -> <name> a, <name> b   (a zero-scaled half is skipped)
    #   chargers/V2G are re-pointed  keep their reference names     (a zero-scaled half is removed)
    idx = {"a": 0, "b": 1}
    for comp, carr, key in (("loads", EV_LOAD, "load"), ("stores", EV_STORE, "store"),
                            ("links", CHARGER, "charger"), ("links", V2G, "v2g")):
        r = set(getattr(ref, comp).index[getattr(ref, comp).carrier == carr])
        s = set(getattr(spl, comp).index[getattr(spl, comp).carrier == carr])
        if comp == "links":
            want = {x for x in r if half_of(x) is None or sc[key][idx[half_of(x)]] > 0}
            what = "re-pointed (names kept)"
        else:
            want = {f"{x} {h}" for x in r for h in HALVES if sc[key][idx[h]] > 0}
            what = "split into a/b"
        add(f"{carr}: reference components {what}", s == want,
            f"missing {sorted(want - s)[:2]}, unexpected {sorted(s - want)[:2]}")

    # chargers/V2G connect LV half x to EV half x (never a<->b)
    cross = []
    for carr in (CHARGER, V2G):
        ev_col, lv_col = ev_port(carr)
        for l in spl.links.index[spl.links.carrier == carr]:
            if half_of(spl.links.at[l, lv_col]) is None or \
                    half_of(spl.links.at[l, lv_col]) != half_of(spl.links.at[l, ev_col]):
                cross.append(l)
    add("Chargers/V2G connect same-letter LV and EV halves", not cross, str(cross[:3]))

    off = list(spl.loads.index[(spl.loads.carrier == EV_LOAD) & ~spl.loads.bus.isin(new_ev)]) + \
        list(spl.stores.index[(spl.stores.carrier == EV_STORE) & ~spl.stores.bus.isin(new_ev)])
    add("EV loads/stores sit on EV half buses", not off, str(off[:3]))

    # ---- demand conservation
    dr, ds = transport_demand(ref), transport_demand(spl)
    bad = {c: (dr[c], ds.get(c, 0.0)) for c in dr
           if c != EV_LOAD and not np.isclose(dr[c], ds.get(c, 0.0), rtol=1e-6)}
    add("Non-EV transport demand unchanged (oil/H2/methanol/...)", not bad,
        str(bad) if bad else ("no non-EV transport loads in this scenario (check is vacuous)"
                              if len(dr) <= 1 else ""))
    add("EV demand conserved", np.isclose(dr.get(EV_LOAD, 0), ds.get(EV_LOAD, 0), rtol=1e-6),
        f"{dr.get(EV_LOAD, 0):.6g} -> {ds.get(EV_LOAD, 0):.6g}")
    dh = demand_by_half(spl)
    tot = dh["a"] + dh["b"]
    want_a = sc["load"][0] / 2
    add("EV demand on half a follows ev_scaling",
        tot > 0 and np.isclose(dh["a"] / tot, want_a, atol=1e-6),
        f"a share {dh['a'] / tot if tot else float('nan'):.4f}, expected {want_a:.4f}")

    # ---- installed capacity conservation + per-half shares
    ir, isp = installed(ref), installed(spl)
    bad = {k: (ir[k], isp[k]) for k in ir if not np.isclose(ir[k], isp[k], rtol=1e-6, atol=1e-9)}
    add("Installed EV capacities conserved (charger/V2G/store)", not bad, str(bad))
    cap = capacity_by_half(spl)
    for key, label in (("store", "EV store e_nom"), ("charger", "BEV charger p_nom"), ("v2g", "V2G p_nom")):
        t = cap[key]["a"] + cap[key]["b"]
        if t <= 0:
            res.append((f"{label} share on half a", INFO, "total is zero -> skipped"))
            continue
        want = sc[key][0] / 2
        add(f"{label} share on half a follows scaling", np.isclose(cap[key]["a"] / t, want, atol=1e-6),
            f"a share {cap[key]['a'] / t:.4f}, expected {want:.4f}")

    # ---- solved?
    solved = getattr(spl, "objective", None)
    add("Split network has a solution", solved is not None and np.isfinite(solved))
    if solved is None:
        return res

    # ---- informational dispatch / utilisation per half
    for carr in (CHARGER, V2G):
        e = link_energy(spl, carr, by_half=True)
        res.append((f"{carr} energy per half", INFO,
                    ", ".join(f"{k} {v:.4g}" for k, v in e.items() if k != "other" or v)))
    for h in HALVES:
        ids = [l for l in spl.links.index[spl.links.carrier == DIST_CARRIER]
               if half_of(spl.links.at[l, "bus1"]) == h and l in spl.links_t.p0.columns]
        if not ids:
            continue
        cap_col = "p_nom_opt" if "p_nom_opt" in spl.links.columns else "p_nom"
        util = (spl.links_t.p0[ids].abs().max() / spl.links.loc[ids, cap_col].replace(0, np.nan)).dropna()
        res.append((f"LV link utilisation half {h}", INFO,
                    f"max {util.max():.3f}, binding (>=0.999): {(util >= 0.999).sum()}/{len(util)}"
                    if len(util) else "no capacity"))

    # ---- equality with reference when neutral
    c_ref, c_spl = lv_constrained(ref), lv_constrained(spl)
    res.append(("LV constraint active (reference, split)", INFO, f"{c_ref}, {c_spl}"))
    if neutral and c_ref == c_spl:
        add("LV distribution links extendable/priced as in reference", dist_sig(ref) == dist_sig(spl),
            f"(extendable, capital_cost) ref {dist_sig(ref)} vs split {dist_sig(spl)}")
        add("Objective equals reference", np.isclose(ref.objective, spl.objective, rtol=rtol),
            f"{ref.objective:.6g} vs {spl.objective:.6g}")

        cr, cs = capacities(ref), capacities(spl)
        bad = {k: (cr[k], cs.get(k, 0.0)) for k in cr
               if not np.isclose(cr[k], cs.get(k, 0.0), rtol=rtol, atol=1e-6)}
        add("Optimal capacities equal reference (EV chain + distribution links)", not bad,
            str({k: (round(a, 3), round(b, 3)) for k, (a, b) in bad.items()}))

        # atol scales with EV demand: V2G throughput is tiny and degenerate, so a
        # relative-only test flags solver noise
        atol = rtol * dr.get(EV_LOAD, 0.0)
        for carr in (CHARGER, V2G):
            if carr == V2G and not has_v2g:
                continue
            er, es = link_energy(ref, carr), link_energy(spl, carr)
            add(f"{carr} energy equals reference (atol = rtol x EV demand)",
                np.isclose(er, es, rtol=rtol, atol=atol),
                f"{er:.6g} vs {es:.6g}, EV demand {dr.get(EV_LOAD, 0.0):.6g}")

        for label, buses in (("LV", lvl), ("EV", new_ev)):
            pr = prices(spl, buses)
            worst = 0.0
            for b in pr.columns:
                if b.endswith(" a") and b[:-2] + " b" in pr.columns:
                    worst = max(worst, float((pr[b] - pr[b[:-2] + " b"]).abs().max()))
            add(f"{label} prices equal on half a and half b",
                worst < 1e-3 * max(1.0, float(pr.abs().max().max())), f"max diff {worst:.3g}")

        if not np.isclose(ref.objective, spl.objective, rtol=rtol):
            res.extend(diagnose(ref, spl, rtol))
    else:
        why = ("LV constraint differs between reference and split" if c_ref != c_spl
               else "non-neutral config")
        res.append(("Equality with reference", INFO, f"skipped: {why}"))
        res.append(("Optimal capacities (informational)", INFO,
                    str({k: round(v, 2) for k, v in capacities(spl).items()})))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("split")
    ap.add_argument("--config")
    ap.add_argument("--rtol", type=float, default=1e-3)
    a = ap.parse_args()
    ref, spl = pypsa.Network(a.ref), pypsa.Network(a.split)
    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    sec = cfg.get("sector", {})
    nested = sec.get("electricity_distribution_grid_split_lv_ev")
    nested = nested if isinstance(nested, dict) else {}
    t = {k: nested.get(k, sec.get(f"electricity_distribution_grid_split_lv_{k}")) for k in SCALING_KEYS}
    res = validate(ref, spl, resolve(t), a.rtol)
    w = max(len(r[0]) for r in res)
    for name, status, detail in res:
        print(f"{status:5s} {name:{w}s}  {detail}")
    sys.exit(1 if any(r[1] == FAIL for r in res) else 0)


if __name__ == "__main__":
    main()