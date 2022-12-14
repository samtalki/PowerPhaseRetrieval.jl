module PowerPhaseRetrieval
include("/home/sam/github/PowerSensitivities.jl/src/PowerSensitivities.jl")
using .PowerSensitivities
using LinearAlgebra

include("core/nr_linear_approx.jl") #taylor series linear approximations
include("prob/ph_retrieve.jl")
include("prob/sens_phret.jl")
include("prob/nr_ph_retrieve.jl")


#--- Phase retrieval problem
export SensitivityPhaseRetrieval
export nr_phase_retrieval,nr_phase_retrieval! #Netwon-Raphson standard phase retrieval
export sdp_sens_phret,maxcut_sens_phret #Senstivitiy matrix phase retrieval.

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

greet() = print("Hello World!")

end # module
