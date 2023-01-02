include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM

net = PM.make_basic_network(PM.parse_file("/home/sam/github/PowerSensitivities.jl/data/radial_test/case14.m"))
PM.compute_ac_pf!(net)
idx = PPR.calc_bus_idx_of_type(net,[1])
Y = PM.calc_basic_admittance_matrix(net)
Yprime = Y[idx,idx]
Vrect = PM.calc_basic_bus_voltage(net)
Vrectprime = Vrect[idx]
S = PM.calc_basic_bus_injection(net)
Sprime = S[idx]

Im = abs.(S) ./ abs.(Vrect)
Imprime = abs.(Sprime) ./ abs.(Vrectprime)

Irect = Y*Vrect
Irectprime = Yprime*Vrectprime
Imprime_true = abs.(Irectprime)


results = PPR.solve_ybus_phasecut!(net)
