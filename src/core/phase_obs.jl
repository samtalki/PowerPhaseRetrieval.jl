"""
Struct for the Δθ observability condition in Theorem 1.
"""
struct PhaseObservability
    lhs::Vector{Float64}
    rhs_row::Vector{Float64}
    rhs_col::Vector{Float64}
    observable::Vector{Bool}
    strong_observable::Vector{Bool}
end


function dqdth_observability(net::Dict;sel_bus_types=[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    q = imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Sqv = J.qv[sel_bus_idx,sel_bus_idx]
    Spv = J.pv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    
    # --- check observability of Δθ using ∂q∂θ condition ---#
    dqth_rhs_col,dqth_rhs_row = [],[]
    dqth_lhs = []
    dqth_observable,dqth_strong_observable = [],[]
    for i=1:n_bus
        push!(dqth_lhs,
            abs(2*p[i] - v[i]*Spv[i,i])
        )
        push!(dqth_rhs_row,
            sum([v[i]*abs(Spv[k,i]) for k =1:n_bus if k!= i])  
        )
        push!(dqth_rhs_col,
            sum([abs(Spv[i,k]*v[k]) for k=1:n_bus if k!=i])
        )
        push!(dqth_observable,
            (dqth_lhs[i] >= dqth_rhs_row[i]) || (dqth_lhs[i] >= dqth_rhs_col[i])
        )
        push!(dqth_strong_observable,
            2*abs(p[i]) + v[i]*abs(Spv[i,i]) >= sum([v[i]*abs(Spv[k,i]) for k=1:n_bus if k!= i])
            || 
            2*abs(p[i]) +  v[i]*abs(Spv[i,i]) >= sum([v[k]*abs(Spv[i,k]) for k=1:n_bus if k!= i])
        )
    end
    
    return PhaseObservability(
        dqth_lhs,dqth_rhs_row,dqth_rhs_col,dqth_observable,dqth_strong_observable
    )   
end
function dpdth_dqdth_observability(net::Dict;sel_bus_types =[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    q = imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Sqv = J.qv[sel_bus_idx,sel_bus_idx]
    Spv = J.pv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    

    
    # --- check observability of Δθ using ∂p∂θ condition ---#
    dpth_rhs_col,dpth_rhs_row = [],[]
    dpth_lhs = []
    dpth_observable,dpth_strong_observable = [],[]
    for i=1:n_bus
        push!(dpth_lhs,
            abs(v[i]*Sqv[i,i] - 2*q[i])
        )
        push!(dpth_rhs_row,
            sum([v[i]*abs(Sqv[k,i]) for k =1:n_bus if k!= i])  
        )
        push!(dpth_rhs_col,
            sum([abs(Sqv[i,k]*v[k]) for k=1:n_bus if k!=i])
        )
        push!(dpth_observable,
            (dpth_lhs[i] >= dpth_rhs_row[i]) || (dpth_lhs[i] >= dpth_rhs_col[i])
        )
        push!(dpth_strong_observable,
            2*abs(q[i]) + v[i]*abs(Sqv[i,i]) >= sum([v[i]*abs(Sqv[k,i]) for k=1:n_bus if k!= i])
            || 
            2*abs(q[i]) +  v[i]*abs(Sqv[i,i]) >= sum([v[k]*abs(Sqv[i,k]) for k=1:n_bus if k!= i])
        )
    end

    # --- check observability of Δθ using ∂q∂θ condition ---#
    dqth_rhs_col,dqth_rhs_row = [],[]
    dqth_lhs = []
    dqth_observable,dqth_strong_observable = [],[]
    for i=1:n_bus
        push!(dqth_lhs,
            abs(2*p[i] - v[i]*Spv[i,i])
        )
        push!(dqth_rhs_row,
            sum([v[i]*abs(Spv[k,i]) for k =1:n_bus if k!= i])  
        )
        push!(dqth_rhs_col,
            sum([abs(Spv[i,k]*v[k]) for k=1:n_bus if k!=i])
        )
        push!(dqth_observable,
            (dqth_lhs[i] >= dqth_rhs_row[i]) || (dqth_lhs[i] >= dqth_rhs_col[i])
        )
        push!(dqth_strong_observable,
            2*abs(p[i]) + v[i]*abs(Spv[i,i]) >= sum([v[i]*abs(Spv[k,i]) for k=1:n_bus if k!= i])
            || 
            2*abs(p[i]) +  v[i]*abs(Spv[i,i]) >= sum([v[k]*abs(Spv[i,k]) for k=1:n_bus if k!= i])
        )
    end

    # --- Check if either condition holds ---#
    observable = dqth_observable .| dpth_observable
    strong_observable = dqth_strong_observable .| dpth_strong_observable
    
    # --- calculate worst radii ---#
    #---- worst radii of the observability condition relative to the left hand side of the complex plane ---#
    worst_radii = []
    for i=1:n_bus
        if !(dqth_strong_observable[i] && dpth_strong_observable[i]) #---- if not observable, how close did we get to the boundary? ---#
            if !(dpth_observable[i]) && !(dqth_observable[i])
                push!(worst_radii,
                    min(dpth_rhs_row[i] - dpth_lhs[i],dpth_rhs_col[i] - dpth_lhs[i],dqth_rhs_row[i] - dqth_lhs[i],dqth_rhs_col[i] - dqth_lhs[i])
                )
            elseif !(dpth_observable[i])
                push!(worst_radii,
                    min(dpth_rhs_row[i],dpth_rhs_col[i]) - dpth_lhs[i]
                )
            elseif !(dqth_observable[i])
                push!(worst_radii,
                    min(dqth_rhs_row[i],dqth_rhs_col[i]) - dqth_lhs[i]
                )
            end
        else
            push!(worst_radii,0.0)
        end
    end
    worst_radius = maximum(worst_radii)


    return worst_radius,PhaseObservability(
        dqth_lhs,dqth_rhs_row,dqth_rhs_col,observable,strong_observable
    )
end