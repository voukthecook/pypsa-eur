import pypsa
n = pypsa.Network("results/D2SplitLV_GR_3_clusters_24H_coupled/networks/base_s_3_lv1.2_Co2L0-24H_2030.nc")

ev = n.links[n.links.carrier.isin(["BEV charger", "V2G"])]
print(ev[["carrier", "bus0", "bus1", "p_nom"]])

ev_load = n.loads[n.loads.carrier == "land transport EV"]
print(ev_load[["bus"]].assign(mean_MW=n.loads_t.p_set[ev_load.index].mean()))