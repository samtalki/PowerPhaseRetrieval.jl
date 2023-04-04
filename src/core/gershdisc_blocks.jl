function is_jacobian_block_invertible(v,p,q,dpv,dqv)
    dpξ,dqξ = calc_augmented_angle_jacobians(v,p,q,dpv,dqv)
    if opnorm(inv(dpξ)*dpv) < 1 && opnorm(inv(dqv)*dqξ) < 1
        return true
    else
        return false
    end
end

function calc_augmented_angle_jacobians(v,p,q,dpv,dqv)
    @assert size(dpv) == size(dqv)
    (n,n) = size(dpv)
    #--- define the augmented phase matrices
    dpξ,dqξ = zeros(n,n),zeros(n,n)
    for i=1:n
        for k=1:n
            if i==k
                dpξ[i,k] = v[i]*dqv[i,k] - 2*q[i]
                dqξ[i,k] = -v[i]*dpv[i,k] + 2*p[i]
            else
                dpξ[i,k] = v[i]*dqv[i,k]
                dqξ[i,k] = -v[i]*dpv[i,k]
            end
        end
    end
    return dpξ,dqξ
end

function is_jacobian_block_invertible(net::Dict)
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,[1])
    n_bus = length(sel_bus_idx)
    v = abs.(calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(calc_basic_bus_injection(net))[sel_bus_idx],imag.(calc_basic_bus_injection(net))[sel_bus_idx]
    J_ = PPR.calc_jacobian_matrix(net,[1])
    dpv,dqv = Matrix(J_.pv),Matrix(J_.qv)
    return is_jacobian_block_invertible(v,p,q,dpv,dqv)
end

function calc_augmented_angle_jacobians(net::Dict)
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,[1])
    n_bus = length(sel_bus_idx)
    v = abs.(calc_basic_bus_voltage(net))[sel_bus_idx]
    p,q = real.(calc_basic_bus_injection(net))[sel_bus_idx],imag.(calc_basic_bus_injection(net))[sel_bus_idx]
    J_ = PPR.calc_jacobian_matrix(net,[1])
    dpv,dqv = Matrix(J_.pv),Matrix(J_.qv)
    return calc_augmented_angle_jacobians(v,p,q,dpv,dqv)
end
