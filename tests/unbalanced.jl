using Plots,LaTeXStrings
using PowerModelsDistribution
using LinearAlgebra
using ForwardDiff
using Ipopt


# --------------------- Form the Jacobian matrices ---------------------
#--- Jacobian of the power flow equations w.r.t. voltage magnitudes and angles,
# "A survey of relaxations and approximations of the power flow equations", Molzahn and Hiskens (2013)
function dsθ(vm,va)
    vph = vm .* cis.(va)
    ∂sθ = im.*(
        Diagonal(vph)*(
            Diagonal( conj.(Y)*conj.(vph) ) - conj.(Y)*Diagonal( conj.(vph) )
        )
    )
    return ∂sθ
end

function dsvm(vm,va)
    vph = vm .* cis.(va)
    ∂svm = Diagonal(vph)*(
        Diagonal( conj.(Y)*conj.(vph) ) + conj.(Y)*Diagonal( conj.(vph) )
    )*Diagonal(1 ./ vm)
    return ∂svm 
end

#----- real and imaginary parts of the Jacobian matrices
dpθ(vm,va)= real.(dsθ(vm,va))
dqθ(vm,va) = imag.(dsθ(vm,va))
dpvm(vm,va) = real.(dsvm(vm,va))
dqvm(vm,va) = imag.(dsvm(vm,va))

#---- test the symmetry hypothesis of the Jacobian matrices
dpξ(vm,va,q) = Diagonal(vm)*dqvm(vm,va) - 2*Diagonal(q) # ∂p/∂ξ
dqξ(vm,va,p) = -Diagonal(vm)*dpvm(vm,va) + 2*Diagonal(p) # ∂q/∂ξ



# Load the data -- case 3 unbalanced
eng = parse_file("data/unbalanced/case3_unbalanced.dss")
math = transform_data_model(eng)

# Load a power flow formulation
pflow_model = instantiate_mc_model(eng, ACPUPowerModel,build_mc_pf)
result = solve_mc_pf(math,ACPUPowerModel,Ipopt.Optimizer)

# ----- build the compound block-admittance matrix for the unbalanced network
K = length(math["bus"]) # number of buses
Y = zeros(ComplexF64,3*K,3*K)

#--------------------- off diagonal admittances ---------------------
for (branch_idx,branch) in math["branch"]
    fr_bus, to_bus = branch["f_bus"], branch["t_bus"]    
    fr_idx, to_idx = math["bus"][string(fr_bus)]["index"], math["bus"][string(to_bus)]["index"]
    print(fr_idx,to_idx)
    G,B = calc_branch_y(branch)
    Y[3*fr_idx-2:3*fr_idx,3*to_idx-2:3*to_idx] -= G + im.*B
    Y[3*to_idx-2:3*to_idx,3*fr_idx-2:3*fr_idx] -= G + im.*B
end

# --------------------- diagonal admittances ---------------------
for (bus_idx,bus) in math["bus"]
    i = bus["index"]
    Y[3*i-2:3*i,3*i-2:3*i] = zeros(ComplexF64,3,3)
    for j=1:K # for each column block
        if j==i
            continue
        else
            Y[3*i-2:3*i,3*i-2:3*i] -= Y[3*i-2:3*i,3*j-2:3*j] # subtract the off-diagonal elements to get the diagonal elements
        end
    end
end

@assert all(Y .≈ conj(Y')) "Y must be symmetric"

# --------------------- Get the power flow solution data ---------------------
vm0,va0 = zeros(Float64,3*K),zeros(Float64,3*K) #voltage magnitudes and angles (3)
p0,q0 = zeros(Float64,3*K),zeros(Float64,3*K) #active and reactive power net injections

#--- voltage magnitude and phase angles
for (bus_id,bus_sol) in result["solution"]["bus"]
    i = math["bus"][bus_id]["index"]
    vm0[3*i-2:3*i] .= bus_sol["vm"]
    va0[3*i-2:3*i] .= bus_sol["va"]
end

#--- active and reactive power net injections

# loads
for (load_id,load_sol) in result["solution"]["load"]
    i = math["load"][load_id]["load_bus"] 
    pd = load_sol["pd_bus"]
    qd = load_sol["qd_bus"]
    p0[3*i-2:3*i] .-= pd
    q0[3*i-2:3*i] .-= qd
end

# generators
for (gen_id,gen_sol) in result["solution"]["gen"]
    i = math["gen"][gen_id]["gen_bus"]
    pg = gen_sol["pg_bus"]
    qg = gen_sol["qg_bus"]
    p0[3*i-2:3*i] .+= pg
    q0[3*i-2:3*i] .+= qg
end

#---- voltage phasors and complex power injections
vph0 = vm0 .* cis.(va0) # voltage phasors
s0 = p0 + im.*q0 # complex power injections

#----- test the symmetry hypothesis of the Jacobian matrices
dpθ_0 = dpθ(vm0,va0)
dqθ_0 = dqθ(vm0,va0)
dpvm_0 = dpvm(vm0,va0)
dqvm_0 = dqvm(vm0,va0)
dpξ_0 = dpξ(vm0,va0,q0)
dqξ_0 = dqξ(vm0,va0,p0)

#----- errors of the symmetry hypothesis of the Jacobian matrices
err_dpθ = norm(dpθ(vm0,va0) - dpξ(vm0,va0,q0))/norm(dpθ(vm0,va0))
err_dqθ = norm(dqθ(vm0,va0) - dqξ(vm0,va0,p0))/norm(dqθ(vm0,va0))