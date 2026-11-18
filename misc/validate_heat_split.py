"""
Validate a solved heat-split network against an unsplit reference run.

Usage:
    python validate_heat_split.py REF.nc SPLIT.nc [--config cfg.yaml] [--rtol 1e-3]

With a neutral config (50/50 shares, scales 1, all techs allowed) the split
network must reproduce the reference: same objective, same capacity and energy
totals per technology, and identical heat prices on half a and half b.
With a non-neutral config only the structural and conservation checks apply.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pypsa
import yaml

# Self-contained: same matching rules as heat_split_block.py in prepare_sector_network.py
HALVES = ("a", "b")
SYSTEMS = ("rural", "urban decentral")
PLURAL = {"Bus": "buses", "Load": "loads", "Generator": "generators",
          "Store": "stores", "Link": "links"}
TECH_PATTERNS = (("water tank", "water tank"), ("heat pump", "heat pump"),
                 ("resistive heater", "resistive heater"),
                 ("solar thermal", "solar thermal"), ("boiler", "boiler"))


def system_of_carrier(carrier):
    c = str(carrier).lower()
    if "urban central" in c:
        return None
    if "urban decentral" in c:
        return "urban decentral"
    if "rural" in c:
        return "rural"
    return None


def _is_chain_carrier(carrier):
    c = str(carrier).lower()
    return system_of_carrier(c) is not None and ("heat" in c or "water tank" in c)


def chain_buses(n):
    return n.buses.index[n.buses.carrier.map(_is_chain_carrier)]


def _tech_of(carrier):
    c = str(carrier).lower()
    for pattern, tech in TECH_PATTERNS:
        if pattern in c:
            return tech
    return "other"


PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


def _norm(v):
    v = (1.0, 1.0) if v is None else tuple(float(x) for x in v)
    t = sum(v)
    return tuple(2 * x / t for x in v) if t > 0 else v


def is_neutral(h):
    return (np.allclose(_norm(h.get("heat_scaling")), 1.0)
            and np.allclose(_norm(h.get("hp_scaling")), 1.0))


def demand_by_half(n):
    """Heat demand energy on half a and half b (split networks only)."""
    w = n.snapshot_weightings.generators
    out = {"a": 0.0, "b": 0.0}
    for i in n.loads.index[n.loads.bus.isin(set(chain_buses(n)))]:
        bus = n.loads.at[i, "bus"]
        h = "a" if bus.endswith(" a") else "b" if bus.endswith(" b") else None
        if h is None:
            continue
        out[h] += float(n.loads_t.p_set[i].mul(w).sum()) if i in n.loads_t.p_set.columns \
            else float(n.loads.at[i, "p_set"] * w.sum())
    return out


def heat_demand(n):
    w = n.snapshot_weightings.generators
    chain = set(chain_buses(n))
    out = {s: 0.0 for s in SYSTEMS}
    for i in n.loads.index[n.loads.bus.isin(chain)]:
        s = system_of_carrier(n.buses.at[n.loads.at[i, "bus"], "carrier"])
        if i in n.loads_t.p_set.columns:
            out[s] += float(n.loads_t.p_set[i].mul(w).sum())
        else:
            out[s] += float(n.loads.at[i, "p_set"] * w.sum())
    return out


def capacities(n):
    out = {}
    for cname, nom in (("Link", "p_nom"), ("Store", "e_nom"), ("Generator", "p_nom")):
        static = getattr(n, PLURAL[cname])
        col = nom + "_opt" if nom + "_opt" in static.columns else nom
        for s in SYSTEMS:
            sub = static[static.carrier.map(system_of_carrier) == s]
            for c, v in sub[col].groupby(sub.carrier.map(_tech_of)):
                out[f"{cname}|{s}|{c}"] = out.get(f"{cname}|{s}|{c}", 0.0) + float(v.sum())
    return out


def hp_electricity(n):
    w = n.snapshot_weightings.generators
    out = {}
    for s in SYSTEMS:
        ids = n.links.index[(n.links.carrier.map(system_of_carrier) == s)
                            & n.links.carrier.str.contains("heat pump")]
        ids = [i for i in ids if i in n.links_t.p0.columns]
        out[s] = float(n.links_t.p0[ids].mul(w, axis=0).sum().sum()) if ids else 0.0
    return out


def heat_prices(n):
    chain = chain_buses(n)
    return n.buses_t.marginal_price[[b for b in chain if b in n.buses_t.marginal_price]]


def validate(ref, spl, hcfg, rtol=1e-3):
    res = []

    def add(name, ok, detail=""):
        res.append((name, PASS if ok else FAIL, detail))

    neutral = is_neutral(hcfg)
    res.append(("Config neutral (scalings 1,1)", INFO, str(neutral)))

    # ---- are there any decentral heat systems to compare at all?
    for label, net in (("reference", ref), ("split", spl)):
        carriers = set(net.buses.carrier)
        heat = sorted(c for c in carriers if "heat" in c)
        recognised = sorted(c for c in carriers if _is_chain_carrier(c))
        res.append((f"Heat bus carriers in {label}", INFO, str(heat)))
        add(f"Decentral heat systems recognised in {label}", bool(recognised),
            "" if recognised else
            "none matched -> heat checks below are EMPTY, not passed; "
            "carrier names probably differ from SYSTEMS in split_heat.py")

    # ---- structural
    ports = [c for c in spl.links.columns if c.startswith("bus") and c[3:].isdigit()]
    orphans = [(p, b) for p in ports for b in spl.links[p]
               if b not in ("", None) and b not in spl.buses.index]
    for comp in ("loads", "generators", "stores"):
        orphans += [(comp, b) for b in getattr(spl, comp).bus if b not in spl.buses.index]
    add("No component attached to a missing bus", not orphans, str(orphans[:3]))

    chain_left = [b for b in chain_buses(ref) if b in spl.buses.index]
    add("Original decentral heat buses removed", not chain_left, str(chain_left[:3]))

    halves = [b for b in chain_buses(spl) if b.endswith((" a", " b"))]
    add("Heat buses on LV halves a/b exist in split network", len(halves) > 0,
        f"{len(halves)} heat buses with a/b suffix")

    dup = spl.links.index[spl.links.index.duplicated()]
    add("No duplicated link names", len(dup) == 0)

    # urban central heat untouched
    central = lambda n: sorted(b for b in n.buses.index if "urban central heat" in b)
    add("Urban central heat buses unchanged", central(ref) == central(spl))

    # ---- demand conservation
    dr, ds = heat_demand(ref), heat_demand(spl)
    add("Reference has decentral heat demand", sum(dr.values()) > 0, str(dr))
    bad = {s_: (dr[s_], ds[s_]) for s_ in SYSTEMS if not np.isclose(dr[s_], ds[s_], rtol=1e-6)}
    add("Heat demand conserved per system", not bad, str(bad))
    dh = demand_by_half(spl)
    tot = dh["a"] + dh["b"]
    want_a = _norm(hcfg.get("heat_scaling"))[0] / 2
    add("Heat demand on half a follows heat_scaling",
        tot > 0 and np.isclose(dh["a"] / tot, want_a, atol=1e-6),
        f"a share {dh['a'] / tot if tot else float('nan'):.4f}, expected {want_a:.4f}")

    # ---- solved?
    solved = getattr(spl, "objective", None)
    add("Split network has a solution", solved is not None and np.isfinite(solved))
    if solved is None:
        return res

    # ---- equality with reference when neutral
    if neutral:
        add("Objective equals reference",
            np.isclose(ref.objective, spl.objective, rtol=rtol),
            f"{ref.objective:.6g} vs {spl.objective:.6g}")

        cr, cs = capacities(ref), capacities(spl)
        bad = {k: (cr[k], cs.get(k, 0.0)) for k in cr
               if not np.isclose(cr[k], cs.get(k, 0.0), rtol=rtol, atol=1e-6)}
        add("Optimal capacity per system/technology equals reference", not bad,
            str({k: (round(a, 3), round(b, 3)) for k, (a, b) in list(bad.items())[:4]}))

        er, es = hp_electricity(ref), hp_electricity(spl)
        bad = {s: (er[s], es[s]) for s in SYSTEMS
               if not np.isclose(er[s], es[s], rtol=rtol, atol=1e-6)}
        add("Heat-pump electricity use equals reference", not bad, str(bad))

        pr = heat_prices(spl)
        worst = 0.0
        for b in pr.columns:
            if b.endswith(" a") and b[:-2] + " b" in pr.columns:
                worst = max(worst, float((pr[b] - pr[b[:-2] + " b"]).abs().max()))
        add("Heat prices equal on half a and half b", worst < 1e-3 * max(1.0, pr.abs().max().max()),
            f"max diff {worst:.3g}")
    else:
        res.append(("Equality with reference", INFO, "skipped: non-neutral config"))
        cs = capacities(spl)
        res.append(("Capacity totals (informational)", INFO,
                    str({k: round(v, 2) for k, v in list(cs.items())[:6]})))
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
    nested = sec.get("electricity_distribution_grid_split_lv_heat")
    nested = nested if isinstance(nested, dict) else {}
    h = {"heat_scaling": nested.get("heat_scaling",
                                    sec.get("electricity_distribution_grid_split_lv_heat_scaling")),
         "hp_scaling": nested.get("hp_scaling",
                                  sec.get("electricity_distribution_grid_split_lv_hp_scaling"))}
    res = validate(ref, spl, h, a.rtol)
    w = max(len(r[0]) for r in res)
    for name, status, detail in res:
        print(f"{status:5s} {name:{w}s}  {detail}")
    sys.exit(1 if any(r[1] == FAIL for r in res) else 0)


if __name__ == "__main__":
    main()