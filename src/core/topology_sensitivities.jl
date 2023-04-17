# --------------------- Form the Jacobian matrices ---------------------
#--- Jacobian of the power flow equations w.r.t. voltage magnitudes and angles,
# "A survey of relaxations and approximations of the power flow equations", Molzahn and Hiskens (2013)


function calc_dsθ(Y,vm,va)
    vph = vm .* cis.(va)
    ∂sθ = im.*(
        Diagonal(vph)*(
            Diagonal( conj.(Y)*conj.(vph) ) - conj.(Y)*Diagonal( conj.(vph) )
        )
    )
    return ∂sθ
end

function calc_dsvm(Y,vm,va)
    vph = vm .* cis.(va)
    ∂svm = Diagonal(vph)*(
        Diagonal( conj.(Y)*conj.(vph) ) + conj.(Y)*Diagonal( conj.(vph) )
    )*Diagonal(1 ./ vm)
    return ∂svm 
end

#----- real and imaginary parts of the Jacobian matrices
calc_dpθ(Y,vm,va)= real.(calc_dsθ(Y,vm,va))
calc_dqθ(Y,vm,va) = imag.(calc_dsθ(Y,vm,va))
calc_dpvm(Y,vm,va) = real.(calc_dsvm(Y,vm,va))
calc_dqvm(Y,vm,va) = imag.(calc_dsvm(Y,vm,va))

#---- test the symmetry hypothesis of the Jacobian matrices
calc_dpξ(vm,va,q) = Diagonal(vm)*calc_dqvm(Y,vm,va) - 2*Diagonal(q) # ∂p/∂ξ
calc_dqξ(vm,va,p) = -Diagonal(vm)*calc_dpvm(Y,vm,va) + 2*Diagonal(p) # ∂q/∂ξ



function calc_topology_sensitivities!(net::Dict{String,Any};sel_bus_types=[1])
    pq_buses = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    compute_ac_pf!(net)

    # ----------------- get the initial values
    v0 = calc_basic_bus_voltage(net)
    vm0 = abs.(calc_basic_bus_voltage(net))
    θ0 = angle.(calc_basic_bus_voltage(net))
    p0 = real.(calc_basic_bus_injection(net))
    q0 = imag.(calc_basic_bus_injection(net))

    # ----------------- get the topology data
    Y = calc_basic_admittance_matrix(net)

    # ----------------- get the sensitivities (topology known)
    dpθ,dqθ,dpvm,dqvm = calc_dpθ(Y,vm0,θ0),calc_dqθ(Y,vm0,θ0),calc_dpvm(Y,vm0,θ0),calc_dqvm(Y,vm0,θ0)

    return dpθ[pq_buses,pq_buses],dqθ[pq_buses,pq_buses],dpvm[pq_buses,pq_buses],dqvm[pq_buses,pq_buses]

end

