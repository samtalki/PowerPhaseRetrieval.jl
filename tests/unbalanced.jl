using Plots,LaTeXStrings
using PowerModelsDistribution
using LinearAlgebra
using ForwardDiff

# Load the data -- case 3 unbalanced
eng = parse_file("data/unbalanced/case3_unbalanced.dss")
math = transform_data_model(eng)

# Load a power flow formulation
pflow_model = instantiate_mc_model(eng, ACPUPowerModel,build_mc_pf)
result = solve_mc_pf(math,ACPUPowerModel,Ipopt.Optimizer)

# build the compound block-admittance matrix for the unbalanced network
K = length(math["bus"]) # number of buses
Y = zeros(ComplexF64,3*K,3*K)

for (branch_idx,branch) in math["branch"]

    fr_bus, to_bus = branch["f_bus"], branch["t_bus"]    
    fr_idx, to_idx = math["bus"][string(fr_bus)]["index"], math["bus"][string(to_bus)]["index"]
    print(fr_idx,to_idx)
    G,B = calc_branch_y(branch)
    Y[3*fr_idx-2:3*fr_idx,3*fr_idx-2:3*fr_idx] -= G + im.*B
end
