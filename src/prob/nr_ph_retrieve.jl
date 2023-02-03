# BSD 3-Clause License

# Copyright (c) 2022, Samuel Talkington
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
using Distributions,Random
using JuMP
using LinearAlgebra
import Ipopt
#include("/home/sam/github/PowerSensitivities.jl/src/sens/nr_sens.jl")

#Set the seed
Random.seed!(2023)



"""
Given a basic network data dict, compute AC power flow solution and retrieve the voltage phases.
"""
function est_bus_voltage_phase!(network::Dict{String,Any};sel_bus_types=[1],sigma_noise=0.25)
    compute_ac_pf!(network)
    #----- Get relevant PQ bus indeces
    sel_bus_idx = calc_bus_idx_of_type(network,sel_bus_types)
    n_bus = length(sel_bus_idx) #Get num_bus
    Y = calc_basic_admittance_matrix(network)[sel_bus_idx,sel_bus_idx]

    # ---------- Compute nominal ground truth values/params
    #-Compute ground truth complex power injections
    rect_s_nom = calc_basic_bus_injection(network)[sel_bus_idx]
    p_nom,q_nom = real.(rect_s_nom),imag.(rect_s_nom)
    f_x_nom = [p_nom;q_nom]

    #-Compute ground truth Jacobians
    J_nom_model = calc_jacobian_matrix(network,sel_bus_types) #PQ buses only
    J_nom = Matrix(J_nom_model.matrix)
    ∂pθ_true,∂qθ_true = J_nom_model.pth,J_nom_model.qth #--- UNKNOWN angle blocks
    ∂pv,∂qv = J_nom_model.pv,J_nom_model.qv

    #-Compute ground truth voltages
    v_rect_nom = calc_basic_bus_voltage(network)[sel_bus_idx]
    vm_nom,va_nom = abs.(v_rect_nom),angle.(v_rect_nom)
    x_nom = [va_nom;vm_nom]
    # ----------

    #----- Compute  observed mismatches and voltage magnitudes.

    #--- Compute noise parameters for the active and reactive power injections.
    sigma_p = mean(abs.(p_nom[abs.(p_nom) .> 0]))*sigma_noise
    sigma_q = mean(abs.(q_nom[abs.(q_nom) .> 0]))*sigma_noise
    d_p,d_q = Normal(0,sigma_p),Normal(0,sigma_q)

    println(network["name"])
    println("sigmap: ",sigma_p,"sigmaq: ",sigma_q)
    #--- Compute a random perturbation around the operating point,
    # noise distributions for p and q:
    f_x_obs = J_nom*x_nom + [rand(d_p,n_bus);rand(d_q,n_bus)]
    p_obs,q_obs = f_x_obs[1:n_bus],f_x_obs[n_bus+1:end]
    vm_obs = (inv(J_nom)*f_x_obs)[n_bus+1:end]

    #----- Check the reasonableness of this linearization
    linearization_error = norm(x_nom - inv(J_nom)*f_x_obs)/norm(x_nom)
    @assert  linearization_error <= 1e-1 "Failure to linearize! Value: "*string(linearization_error)

    #make phase retrieval model
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    #----- Variables and expressions
    #--- Phase VariableS
    @variable(model,θ[1:n_bus])

    #--- Phase jacobian variables
    @variable(model,∂pθ[1:n_bus,1:n_bus])
    @variable(model,∂qθ[1:n_bus,1:n_bus])

    #---- Expressions
    #- Grid state with unknown phase angles.
    x = [θ; vm_obs]
    #- Jacobian matrix
    J = [
        ∂pθ ∂pv;
        ∂qθ ∂qv
    ]
    #- Residual expression
    @variable(model,resid[1:2*n_bus])
    @constraint(model,resid .== J*x .- f_x_obs)

    #----- Constraints
    #Jacobian physics constraints
    for i =1:n_bus
        @constraint(model,
            ∂pθ[i,i] == vm_obs[i]*∂qv[i,i] - 2*q_nom[i]
        )
        @constraint(model,
            ∂qθ[i,i] == -vm_obs[i]*∂pv[i,i] + 2*p_nom[i]
        )
        @constraint(model,
            [k=1:n_bus; k!= i],
            ∂pθ[i,k] == vm_obs[k]*∂qv[i,k]
        )
        @constraint(model,
            [k=1:n_bus; k!=i],
            ∂qθ[i,k] == -vm_obs[k]*∂pv[i,k]
    )
    end

    #----- Objective - min sum of square errors
    @objective(model,Min,sum(resid.^2))

    optimize!(model)

    #construct phasor voltages
    θ_hat = value.(θ)
    v_rect_hat = vm_nom .* (cos.(θ_hat) + sin.(θ_hat) .* im)

    #param covariance
    ∂pθ = value.(∂pθ)
    ∂qθ = value.(∂qθ)
    
    Je = Matrix([
        ∂pθ ∂pv;
        ∂qθ ∂qv
    ])
    #Construct the standard error:
    #se = sqrt(sum((va_nom.- θ_hat).^2)/(length(θ_hat)-2))
    mse = (1/length(resid))*sum(value.(resid).^2)#(1/length(va_nom))*sum((va_nom.- θ_hat).^2)
    se = sqrt.(diag(inv(Je'*Je))*mse)[1:n_bus]#sqrt.(diag(inv(Jph'*Jph )) * mse)#sqrt.(diag(inv(∂pθ' * ∂pθ + ∂qθ' * ∂qθ))*mse)

    #construct covariance of the error
    err_cov = diag((va_nom .- θ_hat)*(va_nom .- θ_hat)')

    #th_std = sqrt.(diag((inv(Je'*Je)*Je')*(va_nom*va_nom')*(Je*inv(Je'*Je))))
    
    

    #Return the results dict
    return Dict(
        "case_name"=>network["name"],
        "sigma_noise"=>sigma_noise,
        "th_hat"=>θ_hat,
        "th_true"=>va_nom,
        "v_rect_hat"=>v_rect_hat,
        "v_rect_true"=>v_rect_nom,
        "dpth"=>value.(∂pθ),
        "dqth"=>value.(∂qθ),
        "th_sq_errs" => (abs.(va_nom .- θ_hat)).^2,
        "th_rel_err"=> norm(va_nom- θ_hat)/norm(va_nom)*100,
        "std_err"=>se ,
        "err_cov"=>err_cov,
       # "th_std="=>th_std,
        "dpth_rel_err"=>norm(value.(∂pθ)- ∂pθ_true)/norm(∂pθ_true)*100,
        "dqth_rel_err"=>norm(value.(∂qθ)- ∂qθ_true)/norm(∂qθ_true)*100,
        "v_rect_rel_err"=> norm(v_rect_nom-v_rect_hat)/norm(v_rect_nom)*100
    )
end


function nr_phase_retrieval!(network::Dict,sigma_noise=0.0001,sel_bus_types=[1])
    #Solve AC power flow
    #data = NRPhaseRetrieval(network,sigma_noise,sel_bus_types)
    compute_ac_pf!(network)
    return nr_phase_retrieval(network,sigma_noise,sel_bus_types)
end

"""
Given a basic network data dict, construct the physics-informed phase retrieval problem.
"""
function nr_phase_retrieval(network::Dict,sigma_noise=0.0001,sel_bus_types=[1])
    #----- Get relevant PQ bus indeces
    sel_bus_idx = calc_bus_idx_of_type(network,sel_bus_types)
    n_bus = length(sel_bus_idx) #Get num_bus
    Y = calc_basic_admittance_matrix(network)[sel_bus_idx,sel_bus_idx]

    #----- Compute nominal ground truth values/params
    #-Compute ground truth complex power injections
    rect_s_nom = calc_basic_bus_injection(network)[sel_bus_idx]
    p_nom,q_nom = real.(rect_s_nom),imag.(rect_s_nom)
    f_x_nom = [p_nom;q_nom]
    
    #-Compute ground truth Jacobians
    J_nom_model = calc_jacobian_matrix(network,sel_bus_types) #PQ buses only
    J_nom = Matrix(J_nom_model.matrix)
    ∂pθ_true,∂qθ_true = J_nom_model.pth,J_nom_model.qth #--- UNKNOWN angle blocks
    ∂pv,∂qv = J_nom_model.pv,J_nom_model.qv

    #-Compute ground truth voltages
    v_rect_nom = calc_basic_bus_voltage(network)[sel_bus_idx]
    vm_nom,va_nom = abs.(v_rect_nom),angle.(v_rect_nom)
    x_nom = [vm_nom;va_nom]
    
    #----- Compute  observed mismatches
    d_noise = Normal(0,sigma_noise)
    f_x_obs = J_nom*x_nom + rand(d_noise,n_bus*2)
    x_obs = inv(J_nom)*f_x_obs
    vm_obs = x_obs[n_bus+1:end]
    p_obs = f_x_obs[1:n_bus]
    q_obs = f_x_obs[n_bus+1:end]

    #--- Compute a random perturbation around the operating point,
    Δx = rand(d_noise,n_bus*2)
    Δθ_true,Δv = Δx[1:n_bus],Δx[n_bus+1:end]
    x_obs = x_nom + Δx
    vm_obs = x_obs[n_bus+1:end]
    f_x_obs = J_nom*Δx
    Δp,Δq = f_x_obs[1:n_bus],f_x_obs[n_bus+1:end]
    p_obs = p_nom + Δp
    q_obs = q_nom + Δq
    #f_x_obs = [p_obs;q_obs]

    #----- Check the reasonableness of this linearization
    linearization_error = norm(x_nom - inv(J_nom)*f_x_obs)/norm(x_nom)
    @assert  linearization_error <= 1e-1 "Failure to linearize! Value: "*string(norm(x_nom - inv(J_nom)*(f_x_obs))/norm(x_nom))

    #make phase retrieval model
    model = Model(Ipopt.Optimizer)

    #----- Variables and expressions
    #--- Variables
    #Complex voltage phase variable
    @variable(model,θ_r[1:n_bus])
    @variable(model,θ_i[1:n_bus])
    #Phase jacobian variables
    @variable(model,∂pθ[1:n_bus,1:n_bus])
    @variable(model,∂qθ[1:n_bus,1:n_bus])
    
    #--- Affine predicted active/reactive power
    @variable(model,p_pred[1:n_bus])
    @variable(model,q_pred[1:n_bus])
    @variable(model,v_pred[1:n_bus])
    @constraint(model,p_pred .== p_obs + ∂pθ*diagm(Δv)*θ_r + ∂pv*diagm(Δv)*θ_i)
    @constraint(model,q_pred .== q_obs + ∂qθ*diagm(Δv)*θ_r + ∂qv*diagm(Δv)*θ_i)
    
    
    #---- Expressions
    u = [θ_r; θ_i]
    
    #- Design matrix
    M = [
        ∂pθ*Diagonal(Δv) ∂pv*Diagonal(Δv); #∂pθ*Diagonal(vm_obs) ∂pv*Diagonal(vm_obs); #
        ∂qθ*Diagonal(Δv) ∂qv*Diagonal(Δv) #∂qθ*Diagonal(vm_obs) ∂qv*Diagonal(vm_obs)
    ]
    
    #- Residual expression
    @variable(model,resid[1:2*n_bus])
    

    #----- Constraints
        
    #Complex voltage phase constraint |θ_i|==1
    @constraint(model,[k=1:n_bus],θ_r[k]^2 + θ_i[k]^2 == 1)

    #Residual expression
    @constraint(model,resid .== M*u .- f_x_obs)

    #Jacobian physics constraints
    for i =1:n_bus
        @constraint(model,
            ∂pθ[i,i] == vm_obs[i]*∂qv[i,i] - 2*q_pred[i]
        )
        @constraint(model,
            ∂qθ[i,i] == -vm_obs[i]*∂pv[i,i] + 2*p_pred[i]
        )
        @constraint(model,
            [k=1:n_bus; k!= i],
            ∂pθ[i,k] == vm_obs[k]*∂qv[i,k]
        )
        @constraint(model,
            [k=1:n_bus; k!=i],
            ∂qθ[i,k] == -vm_obs[k]*∂pv[i,k]
        )
    end
    
  
    #----- Objective
    @objective(model,Min,sum(resid.^2))

    optimize!(model)

    #construct phase
    θ_r_opt,θ_i_opt = value.(θ_r),value.(θ_i)
    θ_opt = atan.(θ_i_opt,θ_r_opt)

    ∂pθ_hat,∂qθ_hat = value.(∂pθ),value.(∂qθ)
    Δθ_hat = -inv(∂pθ_hat)*∂pv*vm_obs + inv(∂pθ_hat)*p_obs
    θ_hat = va_nom + Δθ_hat
    return Dict(
        "th_opt"=>θ_opt,
        "th_true"=>va_nom,
        "delta_th_true"=>Δθ_true,
        "Δθ_hat"=>Δθ_hat,
        "th_hat"=> θ_hat,
        "θrel_err"=> norm(va_nom- θ_hat)/norm(va_nom)*100,
        "Δrel_err"=>norm(Δθ_true - Δθ_hat)/norm(Δθ_true)*100,
        "dpth"=>value.(∂pθ),
        "dpth_rel_err"=>norm(value.(∂pθ)- ∂pθ_true)/norm(∂pθ_true)*100,
        "dqth"=>value.(∂qθ),
        "dqth_rel_err"=>norm(value.(∂qθ)- ∂qθ_true)/norm(∂qθ_true)*100
    )

end



"""
Given a PowerModels basic network dict, 
construct a JuMP model to estimate the bus voltage phase angles 
"""
function model_est_bus_voltage_phase(data::Dict)
    
    s = PM.calc_basic_bus_injection(data)
    v = PM.calc_basic_bus_voltage(data)
    vmag,θ_true = abs.(v),angle.(v)
    p,q = real.(s),imag.(s)

    signs_q = [-1*sign(q_i) for q_i in q]
    signs  = [signs_q ; signs_q]
    #Get Jacobian 
    J = PS.calc_jacobian_matrix(data)

    #Make problem data
    n = length(p)
    A = [J.pv ; J.qv]
    B = [J.pth ; J.qth]
    x = [p ; q]
    Apinv = pinv(Matrix(A))
    diagX = Diagonal(x)

  

    #Find v∈C^n s.t. 
    model = Model(Ipopt.Optimizer)
    @variable(model,θ[1:n])
    @variable(model,ξ[1:2*n]) #Phase to be learned
    @variable(model,t)
    @constraint(model,[t;ξ] in SecondOrderCone())
    @constraint(model,t==n)
    @objective(model,Min,
        (1/2)*transpose((A*vmag + B*θ) - Diagonal(x)*ξ)*((A*vmag + B*θ) - Diagonal(x)*ξ)
    )
   

    # for i in 1:n
    #     @NLconstraint(model,
    #         k[i] == (1/pf[i])*sqrt(1 - pf[i]^2)
    #     )
    # end
    # for i in n+1:2*n
    #     @NLconstraint(model,
    #         k[i] == pf[i]/(sqrt(1-pf[i]^2))
    #     )
    # end
    return model
end



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


