include("../src/PowerPhaseRetrieval.jl")
import PowerModels as PM
import .PowerPhaseRetrieval as PPR

net = PM.make_basic_network(PM.parse_file("/home/sam/Research/PowerFactorGames/data/case9.m"))
PPR.compute_basic_ac_pf!(net)
nr_data = PPR.calc_basic_ac_pf_data!(net,itr_max=100)