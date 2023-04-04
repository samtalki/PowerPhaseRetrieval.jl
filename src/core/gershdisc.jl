"""
Struct for the Δθ observability condition in Theorem 1.
"""
struct PhaseObservability
    lhs::Vector{Float64}
    rhs_row::Vector{Float64}
    rhs_col::Vector{Float64}
    observable::Vector{Bool}
end


function is_gershdisc_invertible(A::AbstractMatrix)
    n = size(A)[1]
    lhs,rhs = [],[]
    for i=1:n
        push!(lhs,abs(A[i,i]))
        push!(rhs,sum([abs(A[k,i]) for k=1:n if k != i]))
    end
    return all(lhs .> rhs)
end

function is_diagonally_dominant(A::AbstractMatrix)
    n = size(A)[1]
    is = true
    for i=1:n
        if sum([abs(A[k,i]) for k=1:n if k != i]) > abs(A[i,i])
            is = false
        end    
    end
    return is
end

row_offdiag_sum(A) = [sum([abs(A[k,i]) for k=1:size(A)[1] if k != i]) for i=1:size(A)[1]]
col_offdiag_sum(A) = [sum([abs(A[i,k]) for k=1:size(A)[1] if k != i]) for i=1:size(A)[1]]

"""
Given a PowerModels network dict, compute the Δθ recovery guarantee bounds.
"""
function calc_phase_observability(net::Dict;sel_bus_types =[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    q = imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Sqv = J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2]
    rhs_col,rhs_row = [],[]
    lhs,observable = [],[]
    for i = 1:n_bus
        push!(lhs,
            #abs(-v[i]*Sqv[i,i] +2*q[i]) + abs(Sqv[i,i])
            abs(v[i]*Sqv[i,i]) + 2*abs.(q[i]) + abs(Sqv[i,i])
        )
        push!(rhs_row,
            #sum([(1+v[k])*abs(Sqv[i,k]) for k =1:n_bus if k!=i])
            (1 + v[i])*sum([abs(Sqv[k,i]) for k=1:n_bus if k != i])
        )
        push!(rhs_col,
            sum([(1+v[k])*abs(Sqv[i,k]) for k =1:n_bus if k!=i])
        )
        push!(observable,
            (lhs[i] >= rhs_row[i]) || (lhs[i] >= rhs_col[i])
            #abs(q[i]) >=  (1/2)*(1+v[i])*(sum([abs(Sqv[k,i]) for k=1:n_bus if k!=i])- abs(Sqv[i,i]))
        )
    end
    return PhaseObservability(
        lhs,rhs_row,rhs_col,observable
    )
end

"""
Given a network dict, determine the jacobian is invertible without the phase angles needed.
"""
function is_jacobian_invertible(net::Dict,sel_bus_types=[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx],imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Spv,Sqv = J.pv[sel_bus_idx,sel_bus_idx],J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    rhs,lhs,observable = [],[],[]
    for i =1:n_bus
        push!(lhs,
            (1+v[i])*(abs(Sqv[i,i] - Spv[i,i])) + 2*(abs(q[i])-abs(p[i]))
        )
        push!(rhs,
            (1+v[i])*(sum(abs(Sqv[k,i] + Spv[k,i]) for k=1:n_bus if k!=i))
        )
        push!(observable,
            lhs[i] >= rhs[i]
        )
    end
    return observable
end

"""
Inequality 1 for the jacobian invertibility conditions
"""
function jacobian_ineq_1(net::Dict,sel_bus_types=[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx],imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Spv,Sqv = J.pv[sel_bus_idx,sel_bus_idx],J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    rhs,lhs,observable = [],[],[]
    for i =1:n_bus
        push!(lhs,
            v[i]*Sqv[i,i] + 2*q[i]
        )
        push!(rhs,
            v[i]*sum(abs(Sqv[k,i]) for k=1:n_bus if k!=i) + sum(abs(Spv[:,i]))    
        )
        push!(observable,
            lhs[i] >= rhs[i]
        )
    end
    return observable
end

"""
Inequality 2 for the jacobian invertibility conditions
"""
function jacobian_ineq_2(net::Dict,sel_bus_types=[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx],imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Spv,Sqv = J.pv[sel_bus_idx,sel_bus_idx],J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    rhs,lhs,observable = [],[],[]
    for i=1:n_bus
        push!(lhs,
            v[i]*abs(Spv[i,i]) + 2*abs(p[i])
        )
        push!(rhs,
            abs(Sqv[i,i]) - sum(abs(Sqv[k,i]) + abs(Spv[k,i]) for k=1:n_bus if k!=i)
        )
        push!(observable,
            lhs[i] >= rhs[i]
        )
    end
    return observable
end

function jacobian_ineqs_hold(net::Dict,sel_bus_types=[1])
    observable1 = jacobian_ineq_1(net,sel_bus_types)
    observable2 = jacobian_ineq_2(net,sel_bus_types)
    return observable1,observable2
end

"""
Inequality 1 for the jacobian invertibility conditions
"""
function jacobian_ineq_1(net::Dict,sel_bus_types=[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx],imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Spv,Sqv = J.pv[sel_bus_idx,sel_bus_idx],J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    rhs1,lhs1 = [],[]
    rhs2,lhs2 =[],[]
    observable = []
    for i =1:n_bus
        push!(lhs1,
            abs(v[i]*Sqv[i,i] - 2*q[i]) - abs(-v[i]*Spv[i,i] + 2*p[i])
        )
        push!(rhs1,
            (1+v[i])*sum([abs(Spv[k,i]) for k=1:n_bus if k!=i])
        )
        push!(lhs2,
            v[i]*Sqv[i,i] + 2*abs(q[i])
        )
        push!(rhs2,
            v[i]*sum(abs(Sqv[k,i]) for k=1:n_bus if k!=i) + sum(abs.(Spv[:,i]))
        )
        push!(observable,
            (lhs1[i] >= rhs1[i]) || (lhs2[i] >= rhs2[i])
        )
    end
    return observable
end


"""
Inequality 2 for the jacobian invertibility conditions
"""
function jacobian_ineq_2(net::Dict,sel_bus_types=[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(PM.calc_basic_bus_injection(net))[sel_bus_idx],imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Spv,Sqv = J.pv[sel_bus_idx,sel_bus_idx],J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2] == size(Spv)[1] == size(Spv)[2]
    rhs1,lhs1,observable = [],[],[]
    rhs2,lhs2 =[],[]
    for i=1:n_bus
        push!(lhs1,
           abs(Sqv[i,i]) 
        )
        push!(rhs1,
            sum(abs.(Spv[:,i])) + sum([abs(Sqv[k,i]) for k=1:n_bus if k!=i])
        )
        push!(lhs2,
            v[i]*abs(Spv[i,i]) + 2*abs(p[i])
        )
        push!(rhs2,
            abs(Sqv[i,i]) - sum(abs(Sqv[k,i]) + abs(Spv[k,i]) for k=1:n_bus if k!=i)
        )
        push!(observable,
            (lhs1[i] >= rhs1[i]) || (lhs2[i] >= rhs2[i])
        )
    end
    return observable
end