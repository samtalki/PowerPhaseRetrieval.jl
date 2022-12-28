include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM

net = PM.make_basic_network(PM.parse_file("/home/sam/github/PowerSensitivities.jl/data/radial_test/case24_ieee_rts.m"))
X_opt,results = PPR.solve_ybus_phasecut!(net)
