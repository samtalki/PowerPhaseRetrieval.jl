import Convex

struct SensitivityPhaseRetrieval
    S::AbstractMatrix #Voltage sensitivity matrix model ∂vp + j ∂vq
    vm::AbstractArray #voltage magnitude measurements
    p::AbstractArray #active power measurements
    q::AbstractArray #reactive power measurement
    va_true::AbstractArray #GROUND TRUTH voltage angles
end

"""
Bus truncation to make sensitivity phase retrieval model
"""
function SensitivityPhaseRetrieval(network;sel_bus_types=[1])
    #Solve AC power flow
    compute_ac_pf!(network)

    #ground truth complex power injections and voltages
    vrect = calc_basic_bus_voltage(network)
    vm,va = abs.(vrect),angle.(vrect)
    s = calc_basic_bus_injection(network)
    p,q = real.(s),imag.(s)
    n_bus = length(vrect)

    #Compute ground truth sensitivitiy
    S_model = calc_voltage_sensitivity_matrix(network)
    
    #Perform bus truncations
    sel_bus_idx = calc_bus_idx_of_type(network,sel_bus_types)
    vm,va,p,q = vm[sel_bus_idx],va[sel_bus_idx],p[sel_bus_idx],q[sel_bus_idx]
    Svp,Svq = S_model.vp[sel_bus_idx,sel_bus_idx], S_model.vq[sel_bus_idx,sel_bus_idx]
    S = Svp + Svq*im #Complex voltage sensitivity matrix

    return SensitivityPhaseRetrieval(
        S,vm,p,q,va
    )

end


"""
Solve sensitivity phase retrieval problem given a network dict
"""
function sdp_sens_phret(network::Dict;sel_bus_types=[1])
    data = SensitivityPhaseRetrieval(network,sel_bus_types=sel_bus_types)
    return sdp_sens_phret(data)
end


"""
``Fast Rank-One Alternating Minimization Algorithm for Phase Retrieval``
Jian-Feng Cai · Haixia Liu · Yang Wang, Journal of Scientific Computing, 2019.
"""
function sdp_sens_phret(data::SensitivityPhaseRetrieval)
    #Retrieve problem data
    S,vm,p,q = data.S,data.vm,data.p,data.q
    n_bus = length(vm)
    
    #Construct M matrix (fixed phase representation)
    M = diagm(vm)*(I(n_bus) - S*pinv(S))*diagm(vm)
    
    # === 
    # Convex implementation
    # ===
    X = Convex.ComplexVariable(n_bus,n_bus) #X = u* conjtranspose(u)
    obj = Convex.inner_product(X,M)
    constraints = [
        diag(X) == 1
        X in :SDP
    ]
    prob = Convex.minimize(obj,constraints)
    Convex.solve!(prob, () -> SCS.Optimizer())
    X = Convex.evaluate(X)
    value = Convex.objective_value(prob)

    # ================
    # JuMP implementation
    # ================
    # Mr,Mi = real.(M),imag.(M)
    # model = Model(SCS.Optimizer)
    # #Make variables and model
    # @variable(model,Xr[1:n_bus,1:n_bus])
    # @variable(model,Xi[1:n_bus,1:n_bus])
    # @constraint(model, [Xr Xi; -Xi Xr]>=0,PSDCone())
    # @constraint(model,[i=1:n_bus],Xr[i,i] + Xi[i,i] == 1) #diag(X) = 1 ⟹ u[i]*conj(u_i) ==1
    # @objective(model,Min,tr([Mr Mi; -Mi Mr]*[Xr Xi; -Xi Xr]))
    # optimize!(model)
    # value = objective_value(model)

    #Process the solution
    # Xr,Xi = value.(Xr),value.(Xi)
    # X = Xr + Xi*im
    
    eigvals,eigvecs = eigen(X)
    rank_sol = length([e for e in eigvals if(abs(real(e))>1e-4)])
    
    #Take the truncated SVD if the solution is not rank 1.
    if rank_sol >1
        X = calc_closest_rank_r(X,1)
        eigvals,eigvecs = eigen(X)
    end

    #Extract the phase
    θ_rect = eigvecs[:,1];
    for i in 1:n_bus
        θ_rect[i] = θ_rect[i]/abs(θ_rect[i])
    end
    θ = atan.( imag.(θ_rect) ./ real.(θ_rect) )

    return Dict(
        "rank_sol" => rank_sol,
        "X" => X,
        "objective_value"=>value,
        "vm_true"=>vm,
        "va_true"=>data.va_true,
        "va_hat" => θ,
        "va_hat_rect" => θ_rect,
    )
end



# ===================================
# Phase cut, max flow, and phase retrieval approach
# ===================================

"""
Phase cut, max flow, and phase retrieval.
"""
function maxcut_sens_phret(network::Dict;sel_bus_types=[1])
    data = SensitivityPhaseRetrieval(network,sel_bus_types=sel_bus_types)
    return maxcut_sens_phret(data)
end


function maxcut_sens_phret(data::SensitivityPhaseRetrieval)
    #Retrieve problem data
    S,vm,p,q = data.S,data.vm,data.p,data.q
    n_bus = length(vm)

    #make phase retrieval model
    model = Model(SCS.Optimizer)

    #Construct M matrix (fixed phase representation)
    M = Diagonal(vm)*(I(n_bus) - S * pinv(S))*Diagonal(vm)
    Mr,Mi = real.(M),imag.(M)

    #Make variables and model
    @variable(model,ur[1:n_bus])
    @variable(model,uim[1:n_bus])
    @constraint(model,[i=1:n_bus],ur[i]^2 + uim[i]^2==1) #Phase on unity circle
    
    @objective(model,Min,transpose([ur;-uim])*M*[ur; uim])
    optimize!(model)
    return Dict(
        "ur" => value.(ur),
        "uim" => value.(uim),
        "ph_rect" => value.(ur) + value.(uim)*im,
        "ph_angle" => atan.(value.(uim) ./ value.(ur)),
        "objective_value"=>objective_value(model),
        "vm_true"=>vm,
        "va_true"=>data.va_true
    )
end