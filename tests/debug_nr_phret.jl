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
Δx = rand(d_noise,n_bus*2)
Δθ_true = Δx[1:n_bus]
Δv =Δx[n_bus+1:end]
f_x_obs = J_nom*Δx

#----- Check the reasonableness of this linearization
linearization_error = norm(Δx - inv(J_nom)*f_x_obs)/norm(Δx)
@assert  linearization_error <= 1e-1 "Failure to linearize! Value: "*string(linearization_error)

#make phase retrieval model
model = Model(Ipopt.Optimizer)

#----- Variables and expressions
#--- Phase Variables
#-Complex voltage phase variable
@variable(model,u_r[1:n_bus])
@variable(model,u_i[1:n_bus])
#-Change in voltage phase angle variable
@variable(model,Δθ[1:n_bus])

#--- Phase jacobian variables
@variable(model,∂pθ[1:n_bus,1:n_bus])
@variable(model,∂qθ[1:n_bus,1:n_bus])

#--- Affine predicted active/reactive power
@variable(model,p_pred[1:n_bus])
@variable(model,q_pred[1:n_bus])
@constraint(model,p_pred .== p_obs + ∂pθ*diagm(Δv)*u_r + ∂pv*diagm(Δv)*u_i)
@constraint(model,q_pred .== q_obs + ∂qθ*diagm(Δv)*u_r + ∂qv*diagm(Δv)*u_i)


#---- Expressions
u = [u_r; u_i]

#- Design matrix
M = [
    ∂pθ*Diagonal(Δv) ∂pv*Diagonal(Δv);  # ∂pθ*Diagonal(vm_obs) ∂pv*Diagonal(vm_obs); 
    ∂qθ*Diagonal(Δv) ∂qv*Diagonal(Δv)  # ∂qθ*Diagonal(vm_obs) ∂qv*Diagonal(vm_obs)   #
]

#- Residual expression
@variable(model,resid[1:2*n_bus])
#Residual expression
@constraint(model,resid .== M*u .- f_x_obs)

#----- Constraints
    
#Complex voltage phase constraint |u_i|==1
@constraint(model,[k=1:n_bus],u_r[k]^2 + u_i[k]^2 == 1)

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
u_hat = value.(u_r) .+ value.(u_i).*im
@assert all([abs(abs(u_i)-1) ≤ 1e-3  for u_i in u_hat]) 
v_rect_hat = Diagonal(vm_nom)*u_hat

#Extract estimated matrices
∂pθ_hat,∂qθ_hat = value.(∂pθ),value.(∂qθ)
θ_hat = -inv(∂pθ_hat)*∂pv*vm_obs + inv(∂pθ_hat)*p_obs
Δθ_hat = -inv(∂pθ_hat)*∂pv*Δv + inv(∂pθ_hat)*Δp #-inv(∂pθ_hat)*∂pv*vm_obs + inv(∂pθ_hat)*p_obs # #


test_results = Dict(
    "th_hat"=> θ_hat,
    "th_true"=>va_nom,
    "Δθ_hat"=>Δθ_hat,
    "delta_th_true"=>Δθ_true,
    "θrel_err"=> norm(va_nom- θ_hat)/norm(va_nom)*100,
    "Δrel_err"=>norm(Δθ_true - Δθ_hat)/norm(Δθ_true)*100,
    "dpth"=>value.(∂pθ),
    "dpth_rel_err"=>norm(value.(∂pθ)- ∂pθ_true)/norm(∂pθ_true)*100,
    "dqth"=>value.(∂qθ),
    "dqth_rel_err"=>norm(value.(∂qθ)- ∂qθ_true)/norm(∂qθ_true)*100,
    "v_rect_rel_err"=> norm(v_rect_nom-v_rect_hat)/norm(v_rect_nom)*100
)