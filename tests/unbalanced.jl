using PowerModelsDistribution
using JuMP,Ipopt
using LinearAlgebra
using ForwardDiff

# Load the data -- case 3 unbalanced
eng = parse_file("data/unbalanced/case3_unbalanced.dss")
math = transform_data_model(eng)

# Load a power flow formulation
pflow_model = instantiate_mc_model(eng, ACPUPowerModel,build_mc_pf)
result = solve_mc_pf(math,ACPUPowerModel,Ipopt.Optimizer)

# build the compound admittance matrix for the network
K = 3 # number of buses
Yabc = zeros(ComplexF64,3*K,3*K)
for (i,branch) in enumerate(math["branch"])
    G,B = calc_branch_y(branch)
    Yabc[3*(branch["f_bus"]-1)+1:3*(branch["f_bus"]),3*(branch["t_bus"]-1)+1:3*(branch["t_bus"])] += branch["y_abc"]
    Yabc[3*(branch["t_bus"]-1)+1:3*(branch["t_bus"]),3*(branch["f_bus"]-1)+1:3*(branch["f_bus"])] += branch["y_abc"]
end
