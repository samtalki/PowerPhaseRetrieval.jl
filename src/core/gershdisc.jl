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

"""
Given a matrix, returns 2 vectors of booleans for each of row/column gershgorin discs that indicates if they do not contain the origin.
"""
function gershdiscs_exclude_origin(A::AbstractMatrix)
    n = size(A)[1]
    lhs = []
    rhs1 = []
    rhs2 = []
    for i=1:n
        push!(lhs,abs(A[i,i]))
        push!(rhs1,sum([abs(A[k,i]) for k=1:n if k != i]))
        push!(rhs2,sum([abs(A[i,k]) for k=1:n if k != i]))
    end
    return lhs .> rhs1, lhs .> rhs2
end

function is_gershdisc_invertible(A::AbstractMatrix)
    n = size(A)[1]
    lhs = []
    rhs1 = []
    rhs2 = []
    for i=1:n
        push!(lhs,abs(A[i,i]))
        push!(rhs1,sum([abs(A[k,i]) for k=1:n if k != i]))
        push!(rhs2,sum([abs(A[i,k]) for k=1:n if k != i]))
    end
    return all(lhs .> rhs1) || all(lhs .> rhs2)
end

function is_row_diagonally_dominant(A::AbstractMatrix)
    n = size(A)[1]
    is = true
    for i=1:n
        if sum([abs(A[k,i]) for k=1:n if k != i]) > abs(A[i,i])
            is = false
        end    
    end
    return is
end

function is_column_diagonally_dominant(A::AbstractMatrix)
    n = size(A)[1]
    is = true
    for i=1:n
        if sum([abs(A[i,k]) for k=1:n if k != i]) > abs(A[i,i])
            is = false
        end    
    end
    return is
end

function is_diagonally_dominant(A::AbstractMatrix)
    return is_row_diagonally_dominant(A) || is_column_diagonally_dominant(A)
end




function dpdth_observability(net::Dict;sel_bus_types =[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    q = imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Sqv = J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2]
    rhs_col,rhs_row = [],[]
    lhs = []
    observable,strong_observable = [],[]
    for i=1:n_bus
        push!(lhs,
            abs(v[i]*Sqv[i,i] - 2*q[i])
        )
        push!(rhs_row,
            sum([v[i]*abs(Sqv[k,i]) for k =1:n_bus if k!= i])  
        )
        push!(rhs_col,
            sum([abs(Sqv[i,k]*v[k]) for k=1:n_bus if k!=i])
        )
        push!(observable,
            (lhs[i] >= rhs_row[i]) || (lhs[i] >= rhs_col[i])
        )
        push!(strong_observable,
            2*abs(q[i]) + v[i]*abs(Sqv[i,i]) >= sum([v[i]*abs(Sqv[k,i]) for k=1:n_bus if k!= i])
            || 
            2*abs(q[i]) +  v[i]*abs(Sqv[i,i]) >= sum([v[k]*abs(Sqv[i,k]) for k=1:n_bus if k!= i])
        )
    end
    return PhaseObservability(
        lhs,rhs_row,rhs_col,observable,strong_observable
    )
end

row_offdiag_sum(A) = [sum([abs(A[k,i]) for k=1:size(A)[1] if k != i]) for i=1:size(A)[1]]
col_offdiag_sum(A) = [sum([abs(A[i,k]) for k=1:size(A)[1] if k != i]) for i=1:size(A)[1]]


function calc_phase_observability(net::Dict;sel_bus_types =[1])
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    n_bus = length(sel_bus_idx)
    J = PPR.calc_jacobian_matrix(net)
    v = abs.(PM.calc_basic_bus_voltage(net))[sel_bus_idx]
    q = imag.(PM.calc_basic_bus_injection(net))[sel_bus_idx]
    Sqv = J.qv[sel_bus_idx,sel_bus_idx]
    @assert length(v) == length(q) == size(Sqv)[1] == size(Sqv)[2]
    rhs_col,rhs_row = [],[]
    lhs,observable,strong_observable = [],[],[]
    for i = 1:n_bus
        push!(lhs,
            #abs(v[i]*Sqv[i,i]) + 2*abs.(q[i]) + abs(Sqv[i,i])
            #abs(Sqv[i,i]) + abs(v[i]*Sqv[i,i] - 2*q[i])
            abs(Sqv[i,i]*v[i] - 2*q[i])
        )
        push!(rhs_row,
            #sum([(1+v[k])*abs(Sqv[i,k]) for k =1:n_bus if k!=i])
            (1 + v[i])*sum([abs(Sqv[k,i]) for k=1:n_bus if k != i])
        )
        push!(rhs_col,
            sum([(1+v[k])*abs(Sqv[i,k]) for k =1:n_bus if k!=i])
        )
        push!(observable,
            #abs(q[i]) >= 0.5*v[i]*(sum([abs(Sqv[k,i]) for k=1:n_bus if k!= i])- abs(Sqv[i,i]))
            (lhs[i] >= rhs_row[i]) || (lhs[i] >= rhs_col[i])
            #abs(q[i]) >=  (1/2)*(1+v[i])*(sum([abs(Sqv[k,i]) for k=1:n_bus if k!=i])- abs(Sqv[i,i]))
        )
        push!(strong_observable,
            #abs(q[i]) >= 0.5*v[i]*(sum([abs(Sqv[k,i]) for k=1:n_bus if k!= i])- abs(Sqv[i,i]))    
            (abs(q[i]) >= 0.5*(1+v[i])*(sum([abs(Sqv[k,i]) for k=1:n_bus if k!=i])- abs(Sqv[i,i]))) || ((1+v[i])*abs(Sqv[i,i]) + 2*abs(q[i])>=sum([(1+v[k])*abs(Sqv[i,k]) for k =1:n_bus if k!=i]))
        )
    end
    return PhaseObservability(
        lhs,rhs_row,rhs_col,observable,strong_observable
    )
end

function calc_phase_observability!(net::Dict) 
    PM.compute_ac_pf!(net) 
    return calc_phase_observability(net)
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
    rhs1,lhs1 = [],[]
    rhs2,lhs2 = [],[]
    observable = zeros(Bool,n_bus)
    #---- row sum condition
    for i =1:n_bus
        push!(lhs1,
            2*(abs(p[i]) + abs(q[i]))
        )
        push!(rhs1,
            v[i]*(abs(Spv[i,i]) - abs(Sqv[i,i])) + (1+v[i])*sum([abs(Spv[k,i]) for k=1:n_bus if k!=i])
        )
        push!(lhs2,
            abs(Sqv[i,i])
        )
        push!(rhs2,
            sum(abs.(Spv[:,i])) + sum(abs(Sqv[k,i]) for k=1:n_bus if k!=i)
        )
        observable[i]= lhs1[i] >= rhs1[i] && lhs2[i] >= rhs2[i] #update observability
    end
    #--- column sum condition
    rhs1,lhs1 = [],[]
    rhs2,lhs2 = [],[]
    for i=1:n_bus
        push!(lhs1,
            2*abs(q[i])
        )
        push!(rhs1,
            sum(abs.(Spv[i,:])) + sum(v[k]*abs(Sqv[i,k]) for k=1:n_bus if k!=i) - v[i]*abs(Sqv[i,i])
        )
        push!(lhs2,
            2*abs(p[i])
        )
        push!(rhs2,
            sum(v[k]*abs(Spv[i,k]) for k=1:n_bus) + sum(abs(Sqv[i,k]) for k=1:n_bus if k!= i) - abs(Sqv[i,i])
        )
        #-- update the observability condition to see if the column sum gives more information than the row sum
        if observable[i] == 0
            observable[i] = lhs1[i] >= rhs1[i] && lhs2[i] >= rhs2[i]
        else
            observable[i] = observable[i]
        end
    end
    return observable
end



#---------
# broken invertibility bounds
#-------

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

