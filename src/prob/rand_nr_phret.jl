using Statistics,Distributions

struct NRPRData
    f_x_nom::Vector # f_x_nom = [p_nom;q_nom]
    x_nom::Vector
    J_nom::PowerFlowJacobian #Nominal Jacobian Matrix
end

function NRPFData(network;sel_bus_types=[1])
    #----- Get relevant PQ bus indeces
    compute_ac_pf!(network)
    sel_bus_idx = calc_bus_idx_of_type(network,sel_bus_types)
    n_bus = length(sel_bus_idx) #Get num_bus
    # ---------- Compute nominal ground truth values/params
    #-Compute ground truth complex power injections
    rect_s_nom = calc_basic_bus_injection(network)[sel_bus_idx]
    p_nom,q_nom = real.(rect_s_nom),imag.(rect_s_nom)
    f_x_nom = [p_nom;q_nom]
    #-Compute ground truth voltages
    v_rect_nom = calc_basic_bus_voltage(network)[sel_bus_idx]
    vm_nom,va_nom = abs.(v_rect_nom),angle.(v_rect_nom)
    x_nom = [va_nom;vm_nom]
    #-Compute ground truth Jacobians
    J_nom = calc_jacobian_matrix(network,sel_bus_types) #PQ buses only
    return NRPFData(f_x_nom,x_nom,J_nom_model)
end


struct RandomNRPRModel
    net::Dict{String,Any}
    n_bus::Int 
    sigma_jac_noise::Union{Real,Float64} #Noise of the jacobian matrix (simulate "estimate" of jacobian)
    sigma_noise::Union{Real,Float64} #Noise for the observed mismatches
    sel_bus_types::Union{AbstractArray{Integer},Integer}
end


"""
Given a basic network data dict, compute AC power flow solution and retrieve the voltage phases
With a simulated "estimated" power-voltage magnitude sensitivites.
"""
function est_stochastic_bus_voltage_phase!(network::Dict{String,Any};sel_bus_types=[1],sigma_noise=0.25,sigma_jac=0.1)
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
    ∂pv_true,∂qv_true = J_nom_model.pv,J_nom_model.qv

    #---------- Simulate "estimated" jacobians -- add random noise
    sigma_pv = mean(abs.(∂pv_true[abs.(∂pv_true) .> 0]))*sigma_jac
    sigma_qv = mean(abs.(∂pv_true[abs.(∂pv_true) .> 0]))*sigma_jac
    dist_pv,dist_qv = Normal(0,sigma_pv),Normal(0,sigma_qv)
    ∂pv,∂qv = ∂pv_true + rand(dist_pv,n_bus,n_bus), ∂qv_true + rand(dist_qv,n_bus,n_bus)

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

    #Return the results dict
    return Dict(
        "case_name"=>network["name"],
        "sigma_noise"=>sigma_noise,
        "th_hat"=>θ_hat,
        "th_true"=>va_nom,
        "v_rect_hat"=>v_rect_hat,
        "v_rect_true"=>v_rect_nom,
        "obs_dpv_rel_err"=> norm(∂pv - ∂pv_true)/norm(∂pv_true)*100,
        "obs_dqv_rel_err"=> norm(∂qv - ∂qv_true)/norm(∂qv_true)*100,
        "dpth"=>value.(∂pθ),
        "dqth"=>value.(∂qθ),
        "th_sq_errs" => (abs.(va_nom .- θ_hat)).^2,
        "th_rel_err"=> norm(va_nom- θ_hat)/norm(va_nom)*100,
        "dpth_rel_err"=>norm(value.(∂pθ)- ∂pθ_true)/norm(∂pθ_true)*100,
        "dqth_rel_err"=>norm(value.(∂qθ)- ∂qθ_true)/norm(∂qθ_true)*100,
        "v_rect_rel_err"=> norm(v_rect_nom-v_rect_hat)/norm(v_rect_nom)*100
    )
end