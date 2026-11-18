import pypsa
n = pypsa.Network("resources/networks/base_s_5_lv1.2_Co2L0-168H_2030.nc")
trafos = n.links[n.links.carrier == "distribution feeder trafo"]
print(trafos[["bus0", "bus1", "p_nom", "p_nom_extendable"]].to_string())