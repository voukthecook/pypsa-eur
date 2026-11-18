import logging, copy
import numpy as np, pandas as pd, pypsa
from scripts.lv_profile_split import apply_lv_profile_split, drop_zero_share_profiles, PROFILES, HEAT_SYSTEMS
import scripts.lv_profile_split_guards as G
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def build():
    n = pypsa.Network(); n.set_snapshots(pd.RangeIndex(6)); n.snapshot_weightings.loc[:] = 3.0
    for c in ["AC","low voltage","home battery","EV battery","urban central heat","rural heat","urban decentral heat",
              "distribution grid","solar rooftop","gas","heat pump","DAC","BEV charger","home battery charger","land transport EV","H"]:
        n.add("Carrier", c)
    n.add("Bus","N0",carrier="AC"); n.add("Bus","N0 gas",carrier="gas")
    n.add("Bus","N0 low voltage",carrier="low voltage"); n.add("Bus","N0 home battery",carrier="home battery")
    n.add("Bus","N0 EV battery",carrier="EV battery")
    for s in ["urban central","rural"]: n.add("Bus",f"N0 {s} heat",carrier=f"{s} heat")
    n.add("Generator","gas gen",bus="N0 gas",carrier="gas",p_nom=1000,marginal_cost=60)
    n.add("Generator","pv util",bus="N0",carrier="solar rooftop",p_nom_extendable=True,capital_cost=40,p_max_pu=[0,.3,.8,.8,.3,0])
    n.add("Generator","N0 rooftop PV",bus="N0 low voltage",carrier="solar rooftop",p_nom_extendable=True,p_nom_max=50,capital_cost=70,p_max_pu=[0,.3,.8,.8,.3,0])
    n.add("Link","gas OCGT",bus0="N0 gas",bus1="N0",carrier="gas",p_nom=500,efficiency=.4)
    n.add("Link","N0 grid",bus0="N0",bus1="N0 low voltage",carrier="distribution grid",p_nom_extendable=True,capital_cost=20)
    n.add("Link","N0 hp",bus0="N0 low voltage",bus1="N0 urban central heat",carrier="urban central heat pump",p_nom_extendable=True,efficiency=3.,capital_cost=30)
    n.add("Link","N0 rural hp",bus0="N0 low voltage",bus1="N0 rural heat",carrier="rural ground heat pump",p_nom_extendable=True,efficiency=3.,capital_cost=30)
    n.add("Link","N0 DAC",bus0="N0 urban central heat",bus1="N0 low voltage",carrier="DAC",p_nom_extendable=True,efficiency=.5,capital_cost=10)
    n.add("Link","N0 BEV charger",bus0="N0 low voltage",bus1="N0 EV battery",carrier="BEV charger",p_nom=100)
    n.add("Store","N0 home battery store",bus="N0 home battery",carrier="home battery",e_nom_extendable=True,capital_cost=5)
    n.add("Link","N0 hb charger",bus0="N0 low voltage",bus1="N0 home battery",carrier="home battery charger",p_nom_extendable=True,capital_cost=3)
    n.add("Link","N0 hb discharger",bus0="N0 home battery",bus1="N0 low voltage",carrier="home battery charger",p_nom_extendable=True,capital_cost=0)
    n.add("Load","N0 elec",bus="N0 low voltage",carrier="AC",p_set=[10,12,14,12,16,18])
    n.add("Load","N0 ev",bus="N0 EV battery",carrier="land transport EV",p_set=[3,3,3,3,3,3])
    n.add("Load","N0 heat uc",bus="N0 urban central heat",carrier="urban central heat",p_set=[8,8,9,9,8,8])
    n.add("Load","N0 heat r",bus="N0 rural heat",carrier="rural heat",p_set=[4,5,6,5,4,4])
    n.loads_t.p_set["N0 elec"] = pd.Series([10,12,14,12,16,18.0])
    n.add("Load","N0 co2",bus="N0",carrier="AC",p_set=0.)
    # a heat sink so rural/urban heat can be met (heat from hp only)
    return n

def cfg(**kw):
    s=dict(zip(PROFILES,(0.4,0.3,0.3)))
    d=dict(enable=True, shares=s, pv_shares=s, ev_shares=s, heat_shares={h:s for h in HEAT_SYSTEMS}); d.update(kw); return d

def solve(n):
    n.optimize(solver_name="highs"); return n.objective

# 1 conservation + neutrality
base=build(); ob=solve(base)
n=build(); n=apply_lv_profile_split(n,cfg()); drop_zero_share_profiles(n,PROFILES,*(dict(zip(PROFILES,(0.4,0.3,0.3))),)*2,{h:dict(zip(PROFILES,(0.4,0.3,0.3))) for h in G.__dict__.get("X",[]) or ["urban central","urban decentral","rural"]},dict(zip(PROFILES,(0.4,0.3,0.3))),pin_profile=None)
os_=solve(n); print("NEUTRAL base",ob,"split",os_,"rel",abs(os_-ob)/ob)
assert abs(os_-ob)/ob<1e-6

# 2 zero share: no zero clones created
n=build(); s0={"urban":0.5,"suburban":0.5,"rural":0.0}
n=apply_lv_profile_split(n,cfg(shares=s0,pv_shares=s0,ev_shares=s0,heat_shares={h:s0 for h in HEAT_SYSTEMS}))
zero=[x for x in list(n.loads.index)+list(n.links.index)+list(n.generators.index)+list(n.stores.index) if x.endswith(" rural")]
print("zero-share clones:",zero); assert not zero

# 3 conservation failure detection
b=G.snapshot_totals(build()); a=dict(b); k=next(k for k in a if k[2]=="p_nom" and a[k]>0); a[k]*=1.01
try: G.check_conservation(b,a); raise SystemExit("MISSED")
except ValueError as e: print("detected:",str(e)[:90])

# 4 strict classification
n=build(); n.add("Carrier","mystery"); n.add("Link","N0 mystery",bus0="N0",bus1="N0 low voltage",carrier="mystery",p_nom=1)
n=apply_lv_profile_split(n,cfg())
try: apply_lv_profile_split(build().__class__() if False else (lambda m:(m.add("Carrier","mystery"),m.add("Link","N0 mystery",bus0="N0",bus1="N0 low voltage",carrier="mystery",p_nom=1),m)[2])(build()),cfg(strict_classification=True)); raise SystemExit("MISSED strict")
except ValueError as e: print("strict raised:",str(e)[:100])
print("ALL OK")

# 5 report files
import tempfile, os, json, glob
n=build(); n=apply_lv_profile_split(n,cfg())
d=tempfile.mkdtemp(); lp=os.path.join(d,"logs","base_prepare_sector_network.log")
removed={"Link":["a","b"]}
paths=G.write_report(lp,removed); print([os.path.basename(str(x)) for x in paths])
r=json.load(open(paths[0])); assert r["n_conserved_totals"]>0 and r["shares"]["general"]["urban"]==0.4 and r["removed_by_drop_zero_share_and_pin"]=={"Link":2}
assert len(open(paths[1]).read().splitlines())>5
print("REPORT OK")

# 6 attr-misleading warnings suppressed and counted
n=build(); n.add("Carrier","x")
n.links["location"]=""; n.generators["unit"]=""
n.links.loc["N0 grid","location"]="N0"
import io
buf=io.StringIO(); h=logging.StreamHandler(buf); logging.getLogger("pypsa.network.transform").addHandler(h)
n=apply_lv_profile_split(n,cfg())
logging.getLogger("pypsa.network.transform").removeHandler(h)
print("leaked misleading warnings:", buf.getvalue().count("warning-attr-misleading"))
assert buf.getvalue().count("warning-attr-misleading")==0
print("ATTR OK")

# 7 port guard: clean split passes, tampering is caught
def split_with_ev(shares_ev=(0.5, 0.3, 0.2)):
    m = build()
    m.add("Carrier", "V2G"); m.add("Carrier", "EV battery")
    m.add("Store", "N0 EV store", bus="N0 EV battery", carrier="EV battery", e_nom=40.0, e_cyclic=True)
    m.add("Link", "N0 V2G", bus0="N0 EV battery", bus1="N0 low voltage", carrier="V2G", p_nom=30.0)
    ev = dict(zip(PROFILES, shares_ev))
    return apply_lv_profile_split(m, cfg(ev_shares=ev)), ev

n, ev = split_with_ev()
summ = G.check_profile_ports(n, PROFILES); print("clean split:", summ); assert summ["errors"] == 0 and summ["ev_links_checked"] == 6
real = G.ev_split_report(n, PROFILES, ev)
assert abs(real["BEV charger"]["urban"] - 0.5) < 1e-12 and abs(real["EV battery"]["rural"] - 0.2) < 1e-12
print("EV realised shares:", {k: {p: round(x, 3) for p, x in v.items()} for k, v in real.items()})

# 7a charger re-pointed to another profile's EV bus
m, _ = split_with_ev()
m.links.at["N0 BEV charger urban", "bus1"] = "N0 EV battery rural"
try: G.check_profile_ports(m, PROFILES); raise SystemExit("MISSED cross-profile charger")
except ValueError as e: print("cross-profile charger caught:", str(e)[:110])

# 7b V2G with two LV ports (no EV port)
m, _ = split_with_ev()
m.links.at["N0 V2G suburban", "bus0"] = "N0 low voltage suburban"
try: G.check_profile_ports(m, PROFILES); raise SystemExit("MISSED V2G with two LV buses")
except ValueError as e: print("V2G port rule caught:", str(e)[:110])

# 7c EV load without a charger
m, _ = split_with_ev()
m.remove("Link", "N0 BEV charger rural")
try: G.check_profile_ports(m, PROFILES); raise SystemExit("MISSED load without charger")
except ValueError as e: print("load without charger caught:", str(e)[:110])

# 7d ordinary heat pump re-pointed across profiles
m, _ = split_with_ev()
m.links.at["N0 hp urban", "bus1"] = "N0 urban central heat suburban"
try: G.check_profile_ports(m, PROFILES); raise SystemExit("MISSED cross-profile heat pump")
except ValueError as e: print("cross-profile heat pump caught:", str(e)[:110])

# 7e EV capacity not following ev_shares
m, ev = split_with_ev()
m.links.at["N0 BEV charger urban", "p_nom"] *= 1.1
try: G.ev_split_report(m, PROFILES, ev); raise SystemExit("MISSED ev share drift")
except ValueError as e: print("ev share drift caught:", str(e)[:110])

# 7f port check still passes after zero-share drop + pin
m = build(); s0 = {"urban": 0.6, "suburban": 0.4, "rural": 0.0}
m = apply_lv_profile_split(m, cfg(shares=s0, pv_shares=s0, ev_shares=s0, heat_shares={h: s0 for h in HEAT_SYSTEMS}))
rem = drop_zero_share_profiles(m, PROFILES, s0, s0, {h: s0 for h in HEAT_SYSTEMS}, s0, pin_profile="urban")
print("after drop+pin:", G.check_profile_ports(m, PROFILES))

# 7g unsplit LV bus is reported
m = build()
try: G.check_profile_ports(m, PROFILES); raise SystemExit("MISSED unsplit buses")
except ValueError as e: print("unsplit caught:", str(e)[:90])
print("PORT GUARD OK")


# 8 heat-system prefix: zero-share 'residential ...' heat bus is dropped, no orphans remain
def res_net():
    m = build()
    m.add("Carrier", "residential rural heat"); m.add("Carrier", "residential urban decentral heat")
    m.buses.at["N0 rural heat", "carrier"] = "residential rural heat"
    return m

# rural heat system only on urban+suburban; urban central/decentral uniform
s_all = dict(zip(PROFILES, (0.4, 0.3, 0.3)))
s_r = {"urban": 0.6, "suburban": 0.4, "rural": 0.0}
heat = {"urban central": s_all, "urban decentral": s_all, "rural": s_r}
m = apply_lv_profile_split(res_net(), cfg(heat_shares=heat))
removed = drop_zero_share_profiles(m, PROFILES, s_all, s_all, heat, s_all, pin_profile=None)
left = [b for b in m.buses.index if b.endswith("rural heat rural")]
print("zero-share residential heat bus left after drop_zero_share_profiles:", left)
orph = G.orphan_split_buses(m, PROFILES)
print("orphan split buses:", orph)
assert not orph
assert not left, "BUG: residential-prefixed zero-share heat bus survives"
print("PREFIX OK")