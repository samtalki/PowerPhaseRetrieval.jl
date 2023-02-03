"""
Given a PowerModels network and a phase retrieval model, 
Solves the power flow equations with Newton-Raphson power flow,
Solving the phase retrieval problem at every iteration.
"""
function calc_phaseless_nr_pf!(net::Dict{String,Any};tol=1e-4,itr_max=50)
    nr_data = calc_nr_pf!(net,tol=tol,itr_max=itr_max)
    sel_bus_idx = calc_bus_idx_of_type(net,[1])
    return calc_phaseless_nr_pf(nr_data,sel_bus_idx)
end

function calc_phaseless_nr_pf(nr_data::NRPFData,sel_bus_idx)
   # println("in")
   # println(nr_data.f)
    f,x,S,L = nr_data.f,nr_data.x,nr_data.S_inj,nr_data.L_inj
    jacs = nr_data.jacs
    delta_vm = nr_data.delta_vm
    results= []
    for (k,(fk,xk,Sk,Lk,jack)) in enumerate(zip(f,x,S,L,jacs))
        Δpk,Δqk = real.(fk)[sel_bus_idx],imag.(fk)[sel_bus_idx]
        vk,θk = abs.(xk)[sel_bus_idx],angle.(xk)[sel_bus_idx]
        pk,qk = real.(Sk)[sel_bus_idx],imag.(Sk)[sel_bus_idx]
        ∂pvk,∂qvk = jack.pv[sel_bus_idx,sel_bus_idx],jack.qv[sel_bus_idx,sel_bus_idx]
        ∂pθ_true,∂qθ_true = jack.pth[sel_bus_idx,sel_bus_idx],jack.qth[sel_bus_idx,sel_bus_idx]
        push!(results,est_bus_voltage_phase(θk,∂pvk,∂qvk,vk,[pk;qk],qk,pk,∂pθ_true,∂qθ_true)) 
    end
    return results
end

function est_bus_voltage_phase(va_nom,∂pv,∂qv,vm_obs,f_x_obs,q_nom,p_nom,∂pθ_true,∂qθ_true)
    n_bus = length(va_nom)
    #make phase retrieval model
    println("test")
    model = Model(Ipopt.Optimizer)
    #set_silent(model)

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
    v_rect_hat = vm_obs .* (cos.(θ_hat) + sin.(θ_hat) .* im)

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
        "th_hat"=>θ_hat,
        "th_true"=>va_nom,
        "v_rect_hat"=>v_rect_hat,
        "dpth"=>value.(∂pθ),
        "dqth"=>value.(∂qθ),
        "th_sq_errs" => (abs.(va_nom .- θ_hat)).^2,
        "th_rel_err"=> norm(va_nom- θ_hat)/norm(va_nom)*100,
        "std_err"=>se ,
        "err_cov"=>err_cov,
        # "th_std="=>th_std,
        "dpth_rel_err"=>norm(value.(∂pθ)- ∂pθ_true)/norm(∂pθ_true)*100,
        "dqth_rel_err"=>norm(value.(∂qθ)- ∂qθ_true)/norm(∂qθ_true)*100,
        #"v_rect_rel_err"=> norm(v_rect_nom-v_rect_hat)/norm(v_rect_nom)*100
    )
end