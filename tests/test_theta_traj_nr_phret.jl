include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
using PowerModels
using LinearAlgebra
using Random,Distributions
import SCS
import Ipopt

#----- Debug parameters
#rts test case
network = make_basic_network(parse_file("/home/sam/github/PowerSensitivities.jl/data/pm_matpower/case30.m"))
compute_ac_pf!(network)
sel_bus_types = [1]
sigma_noise=0.01

#----- Get relevant PQ bus indeces
sel_bus_idx = PPR.calc_bus_idx_of_type(network,sel_bus_types)
n_bus = length(sel_bus_idx) #Get num_bus
Y = calc_basic_admittance_matrix(network)[sel_bus_idx,sel_bus_idx]

# ---------- Compute nominal ground truth values/params
#-Compute ground truth complex power injections
rect_s_nom = calc_basic_bus_injection(network)[sel_bus_idx]
p_nom,q_nom = real.(rect_s_nom),imag.(rect_s_nom)
f_x_nom = [p_nom;q_nom]

#-Compute ground truth Jacobians
J_nom_model = PPR.calc_jacobian_matrix(network,sel_bus_types) #PQ buses only
J_nom = Matrix(J_nom_model.matrix)
∂pθ_true,∂qθ_true = J_nom_model.pth,J_nom_model.qth #--- UNKNOWN angle blocks
∂pv,∂qv = J_nom_model.pv,J_nom_model.qv

#-Compute ground truth voltages
v_rect_nom = calc_basic_bus_voltage(network)[sel_bus_idx]
vm_nom,va_nom = abs.(v_rect_nom),angle.(v_rect_nom)
x_nom = [va_nom;vm_nom]
# ----------


#----- Compute  observed mismatches
#- Compute a random perturbation around the operating point,
d_noise = Normal(0,sigma_noise) #noise distribution
f_x_obs = J_nom*x_nom + rand(d_noise,n_bus*2)

#----- Check the reasonableness of this linearization
linearization_error = norm(x_nom - inv(J_nom)*f_x_obs)/norm(x_nom)
@assert  linearization_error <= 1e-1 "Failure to linearize! Value: "*string(linearization_error)

#make phase retrieval model
model = Model(Ipopt.Optimizer)

#----- Variables and expressions
#--- Phase Variables
#-Change in voltage phase angle variable
@variable(model,θ[1:n_bus])

#--- Phase jacobian variables
@variable(model,∂pθ[1:n_bus,1:n_bus])
@variable(model,∂qθ[1:n_bus,1:n_bus])

#---- Expressions
x = [θ; vm_nom]

#- Jacobian matrix
J = [
    ∂pθ ∂pv;
    ∂qθ ∂qv
]

#- Residual expression
@variable(model,resid[1:2*n_bus])
#Residual expression
@constraint(model,resid .== J*x .- f_x_obs)

#----- Constraints
#Jacobian physics constraints
for i =1:n_bus
    @constraint(model,
        ∂pθ[i,i] == vm_nom[i]*∂qv[i,i] - 2*q_nom[i]
    )
    @constraint(model,
        ∂qθ[i,i] == -vm_nom[i]*∂pv[i,i] + 2*p_nom[i]
    )
    @constraint(model,
        [k=1:n_bus; k!= i],
        ∂pθ[i,k] == vm_nom[k]*∂qv[i,k]
    )
    @constraint(model,
        [k=1:n_bus; k!=i],
        ∂qθ[i,k] == -vm_nom[k]*∂pv[i,k]
    )
end


#----- Objective
@objective(model,Min,sum(resid.^2))

optimize!(model)

#construct phasor voltages
θ_hat = value.(θ)
v_rect_hat = vm_nom .* (cos.(θ_hat) + sin.(θ_hat) .* im)

#Extract estimated matrices
∂pθ_hat,∂qθ_hat = value.(∂pθ),value.(∂qθ)

test_results = Dict(
    "th_hat"=>θ_hat,
    "th_true"=>va_nom,
    "th_rel_err"=> norm(va_nom- θ_hat)/norm(va_nom)*100,
    "dpth_rel_err"=>norm(value.(∂pθ)- ∂pθ_true)/norm(∂pθ_true)*100,
    "dqth_rel_err"=>norm(value.(∂qθ)- ∂qθ_true)/norm(∂qθ_true)*100,
    "v_rect_rel_err"=> norm(v_rect_nom-v_rect_hat)/norm(v_rect_nom)*100
)