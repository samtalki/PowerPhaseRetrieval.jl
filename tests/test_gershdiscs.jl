include("../src/PowerPhaseRetrieval.jl")
include("../src/core/gershdisc_blocks.jl")
include("../src/core/gershdisc.jl")
include("../src/core/phase_obs.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM
using PowerModels
using LinearAlgebra

# 
path = "/home/sam/Research/PowerPhaseRetrieval/data/networks/"
cases = readdir(path)

# Test the Gershgorin disc theorems
case_results = Dict()

#---- dpdth and dqdth individual recoverability
dpth_phase_recoverable = Dict()
dqth_phase_recoverable = Dict()
pct_dpth_phase_recoverable = Dict()
pct_dqth_phase_recoverable = Dict()
pct_dpth_strong_phase_recoverable = Dict()
pct_dqth_strong_phase_recoverable = Dict()

#---- total phase recoverability
phase_recoverable = Dict()
strong_phase_recoverable = Dict()
pct_phase_recoverable = Dict()
pct_strong_phase_recoverable = Dict()


jac_recoverable = Dict()
pct_jac_recoverable = Dict()
worst_lhs_radii = Dict()
n_pq_bus = Dict()


for case in cases
    # load case and run power flow
    net = PM.make_basic_network(PM.parse_file(path*case))
    PM.compute_ac_pf!(net)
    println("run acpf: $case")

    
    # make a results dict for this case
    case_results[case] = Dict()

    # check if the voltage phase angles are observable
    J = PPR.calc_jacobian_matrix(net,[1])
    dpv,dqv = Matrix(J.pv),Matrix(J.qv)
    dpξ,dqξ = calc_augmented_angle_jacobians(net)


    #--- matrices ---#
    case_results[case]["dpξ"] = dpξ
    case_results[case]["dqξ"] = dqξ
    case_results[case]["dpv"] = dpv
    case_results[case]["dqv"] = dqv

    #-- dpth and dqth individual recoverability ---#
    dpth_phase_recoverable[case] = dpdth_observability(net)
    dqth_phase_recoverable[case] = dqdth_observability(net)

    #--- full observability conditions ---#
    n_pq_bus[case] = size(dpξ)[1]
    worst_radius,ph_rec = dpdth_dqdth_observability(net)
    worst_lhs_radii[case] = worst_radius
    case_results[case]["phase_recoverable"] = ph_rec
    case_results[case]["jac_recoverable"] = is_jacobian_block_invertible(net)



    #--- compute the percentange of singular values that are recoverable ---#
    # --- compute the singular values of the matrices of the corresponding conditions

    #--- theorem 2, assumption 1
    case_results[case]["assum_holds"] = all([rank(dpξ) == size(dpξ)[1],rank(dqv) == size(dqv)[1]])

    #--- theorem 2, condition one
    Σ11 = svdvals(inv(dpξ)*dpv)
    Σ12 = svdvals(inv(dqv)*dqξ)
    Σ11_leq_1 = sum(Σ11 .<= 1)
    Σ12_leq_1 = sum(Σ12 .<= 1)
    case_results[case]["Σ11"] = Σ11
    case_results[case]["Σ12"] = Σ12
    case_results[case]["Σ11_leq_1"] = Σ11_leq_1
    case_results[case]["Σ12_leq_1"] = Σ12_leq_1
    case_results[case]["pct_Σ11_leq1"] = Σ11_leq_1/length(Σ11)
    case_results[case]["pct_Σ12_leq_1"] = Σ12_leq_1/length(Σ12)

    #--- theorem 2, condition two
    Σ21 = svdvals(inv(dpξ)*dqξ)
    Σ22 = svdvals(inv(dqv)*dpv)
    Σ21_leq_1 = sum(Σ21 .<= 1)
    Σ22_leq_1 = sum(Σ22 .<= 1)
    case_results[case]["Σ21"] = Σ21
    case_results[case]["Σ22"] = Σ22
    case_results[case]["Σ21_leq_1"] = Σ21_leq_1
    case_results[case]["Σ22_leq_1"] = Σ22_leq_1
    case_results[case]["pct_Σ21_leq_1"] = Σ21_leq_1/length(Σ21)
    case_results[case]["pct_Σ22_leq_1"] = Σ22_leq_1/length(Σ22)

    #--- save the largest singular value of the two conditions ---#
    case_results[case]["sigma_max_1"] = maximum([Σ11;Σ12])
    case_results[case]["sigma_max_2"] = maximum([Σ21;Σ22])

    # save the recoverabilities
    phase_recoverable[case] = case_results[case]["phase_recoverable"].observable
    strong_phase_recoverable[case] = case_results[case]["phase_recoverable"].strong_observable
    pct_phase_recoverable[case] = sum(phase_recoverable[case])/length(phase_recoverable[case])
    pct_strong_phase_recoverable[case] = sum(strong_phase_recoverable[case])/length(strong_phase_recoverable[case])
    jac_recoverable[case] = case_results[case]["jac_recoverable"]

    # save the dpth and dqth individual recoverabilities
    pct_dpth_phase_recoverable[case] = sum(dpth_phase_recoverable[case].observable)/(n_pq_bus[case])
    pct_dqth_phase_recoverable[case] = sum(dqth_phase_recoverable[case].observable)/(n_pq_bus[case])
    pct_dpth_strong_phase_recoverable[case] = sum(dpth_phase_recoverable[case].strong_observable)/(n_pq_bus[case])
    pct_dqth_strong_phase_recoverable[case] = sum(dqth_phase_recoverable[case].strong_observable)/(n_pq_bus[case])

end

pct_block_phase_recoverable = Dict()
pct_block_strong_phase_recoverable = Dict()

for case in cases
    pct_block_phase_recoverable[case]= max(pct_dpth_phase_recoverable[case],pct_dqth_phase_recoverable[case])
    pct_block_strong_phase_recoverable[case] =  max(pct_dpth_strong_phase_recoverable[case],pct_dqth_strong_phase_recoverable[case])
    
    dpth_lhs,dpth_rhs_col,dpth_rhs_row = dpth_phase_recoverable[case].lhs,dpth_phase_recoverable[case].rhs_col,dpth_phase_recoverable[case].rhs_row
    dqth_lhs,dqth_rhs_col,dqth_rhs_row = dqth_phase_recoverable[case].lhs,dqth_phase_recoverable[case].rhs_col,dqth_phase_recoverable[case].rhs_row
    
    w_pth = [min(dpth_lhs_i - dpth_rhs_col_i,dpth_lhs_i - dpth_rhs_row_i) for (dpth_lhs_i,dpth_rhs_col_i,dpth_rhs_row_i) in zip(dpth_lhs,dpth_rhs_col,dpth_rhs_row)]
    w_qth = [min(dqth_lhs_i - dqth_rhs_col_i,dqth_lhs_i - dqth_rhs_row_i) for (dqth_lhs_i,dqth_rhs_col_i,dqth_rhs_row_i) in zip(dqth_lhs,dqth_rhs_col,dqth_rhs_row)]
    worst_case_radii[case] = maximum(min(w_pth,w_qth))

end

# worst_case_radii = Dict()
# for case in cases
#     #--- get the observability struct---#
#     dpo = dpth_phase_recoverable[case]
#     dqo = dqth_phase_recoverable[case]

#     #--- get the worst case radii ---#
#     worst_radii = []

#     dpth_lhs,dpth_rhs_col,dpth_rhs_row = dpth_phase_recoverable[case].lhs,dpth_phase_recoverable[case].rhs_col,dpth_phase_recoverable[case].rhs_row
#     dqth_lhs,dqth_rhs_col,dqth_rhs_row = dqth_phase_recoverable[case].lhs,dqth_phase_recoverable[case].rhs_col,dqth_phase_recoverable[case].rhs_row
#     WPTH,WQTH = [],[]
#     for i=1:length(dpth_lhs)
#         if dpo.strong_observable[i] 
#             push!(WPTH,0)
#         elseif dqo.strong_observable[i]
#             push!(WQTH,0)
#         elseif !(dpo.strong_observable[i])
#             push!(worst_radii,
#                 min(dpth_rhs_row[i],dpth_rhs_col[i]) - dpth_lhs[i]
#             )    
        
        
#         if !(dpth_phase_recoverable[case].strong_observable[i])
#             push!(worst_radii,
#                 min(dpth_rhs_row[i],dpth_rhs_col[i]) - dpth_lhs[i]
#             )
#         elseif !(dqth_phase_recoverable[case].strong_observable[i])
#             push!(worst_radii,
#                 min(dqth_rhs_row[i],dqth_rhs_col[i]) - dqth_lhs[i]
#             )
#         end
#         w_pth = min(dpth_rhs_row[i],dpth_rhs_col[i]) - dpth_lhs[i]
#         w_qth = min(dqth_rhs_row[i],dqth_rhs_col[i]) - dqth_lhs[i]
#         push!(WPTH,w_pth)
#         push!(WQTH,w_qth)
#     end
#     worst_radii = min.(WPTH,WQTH)
#     worst_radius = maximum(worst_radii)
#     worst_case_radii[case] = worst_radius
# end


#--- save the worst case radii ---#
worst_case_radii = Dict()
2+2==4