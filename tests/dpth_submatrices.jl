include("../src/PowerPhaseRetrieval.jl")
include("../src/core/gershdisc_blocks.jl")
include("../src/core/gershdisc.jl")
include("../src/core/phase_obs.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM
using PowerModels
using LinearAlgebra

#--- solve model and get pq buses
net = make_basic_network(parse_file("/home/sam/Research/PowerPhaseRetrieval/data/networks/case_RTS_GMLC.m"))
pq_buses = PPR.calc_bus_idx_of_type(net,[1])
compute_ac_pf!(net)


#--- get the state data
vph = calc_basic_bus_voltage(net)[pq_buses]
s = calc_basic_bus_injection(net)[pq_buses]
vm,va = abs.(vph),angle.(vph)
p,q = real.(s),imag.(s)


#--- get the jacobian data
J = PPR.calc_jacobian_matrix(net,[1])
dpv,dqv = Matrix(J.pv),Matrix(J.qv)
dpξ,dqξ = calc_augmented_angle_jacobians(net)


#--- get the fixed point injection data
fx_fp = [dpξ dpv; dqξ dqv]*[va;vm]
p_fp,q_fp = fx_fp[1:length(pq_buses)],fx_fp[length(pq_buses)+1:end]

#--- get the expression data
rhs1,rhs2 = inv(dpξ)*(p_fp - dpv*vm),inv(dqξ)*(q_fp - dqv*vm)
@assert norm(rhs1-rhs2) < 1e-6
@assert norm(p_fp - dpξ*rhs1 - dpv*vm) < 1e-6
@assert norm(q_fp - dqξ*rhs1 - dqv*vm) < 1e-6
@assert norm(p_fp - dpξ*rhs2 - dpv*vm) < 1e-6
@assert norm(q_fp - dqξ*rhs2 - dqv*vm) < 1e-6


#--- get the gershgorin discs
center_pth,radii_pth = calc_gershgorin_discs(dpξ)
center_qth,radii_qth = calc_gershgorin_discs(dqξ)

#--- get the gershgorin discs for the submatrices

"""
Constructs a system Xθ where X contains principle rows and columns of the phase angle jacobians
"""
function make_partitioned_system(pth_indxs::Set,qth_idxs::Set;rhs_pth=rhs1,rhs_qth=rhs2)
    @assert isempty(intersect(pth_indxs,qth_idxs))
    n = length(pth_indxs) + length(qth_idxs)
    # --- make design matrix
    X = zeros(n,n)
    for i=1:n
        if i∈pth_indxs
            X[:,i] = dpξ[:,i]
            X[i,:] = dpξ[i,:]
        elseif i∈qth_indxs
            X[:,i] = dqξ[:,i]
            X[i,:] = dqξ[i,:]
        end
    end
    # --- make rhs
    b = zeros(n)
    for i=1:n
        if i∈pth_indxs
            b[i] = rhs_pth[i]   
        elseif i∈qth_indxs
            b[i] = rhs_qth[i]
        end
    end
    return X,b
end
