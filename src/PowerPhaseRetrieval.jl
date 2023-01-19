module PowerPhaseRetrieval
include("/home/sam/github/PowerSensitivities.jl/src/PowerSensitivities.jl")
using .PowerSensitivities
using Distributions,Random,Statistics


include("core/nr_basic.jl")
include("core/nr_data.jl")
#include("core/nr_linear_approx.jl") #taylor series linear approximations
include("core/nr_sens.jl")
include("core/structs.jl") #Phase retrieval models
include("prob/ph_retrieve.jl")
include("prob/sens_phret.jl")
include("prob/nr_ph_retrieve.jl")
include("prob/rand_nr_phret.jl")
include("prob/ybus_phret.jl")
include("prob/ybus_phmax.jl")
include("io/load_samples.jl")


"""
Find the closest rank-R approximate matrix of A
"""
function calc_closest_rank_r(A::Matrix,r::Integer)
    (m,n) = size(A)
    U,Σ,V = svd(A)
    for (i,s_i) in enumerate(Σ)
        if i > r 
            Σ[i] = 0
        end
    end
    return U * Diagonal(Σ) * V' 
end

#--- Newton Raphson
export compute_basic_ac_pf!
export calc_basic_ac_pf_data!

#----- Phase retrieval problem(s)
export SensitivityPhaseRetrieval
export nr_phase_retrieval,nr_phase_retrieval! #Netwon-Raphson standard phase retrieval
export sdp_sens_phret,maxcut_sens_phret #Senstivitiy matrix phase retrieval.
#--- Ybus phasecut
export YbusPhretData,YbusPhretSolution
export solve_ybus_phasecut!
#export solve_greedy_ybus_phasecut,solve_greedy_ybus_phasecut!
#export solve_pq_ybus_phasecut

#--- Jacobian-like matrix utilities
#- Classical AC Power flow Jacobian utilities
export PowerFlowJacobian, calc_jacobian_matrix
export calc_pth_jacobian, calc_qth_jacobian
export calc_pv_jacobian, calc_qv_jacobian
#- Voltage Sensitivity Matrix (underdetermined inverse AC Power Flow Jacobian) utilities
export VoltageSensitivityMatrix
export calc_voltage_sensitivity_matrix
#- Spectral Analysis utilities
export SpectralAnalysis
export calc_spectral_analysis,calc_condition_number
#- Timeseries data utilities
export AMIData
export EstimatedJacobians

end # module
