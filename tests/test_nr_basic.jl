include("../src/PowerPhaseRetrieval.jl")
import PowerModels as PM
import .PowerPhaseRetrieval as PPR

net = PM.make_basic_network(PM.parse_file("data/case_RTS_GMLC.m"))
nr_data = PPR.calc_nr_pf!(net,itr_max=100)
results = PPR.calc_phaseless_nr_pf!(net)