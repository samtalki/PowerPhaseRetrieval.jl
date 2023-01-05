


function solve_ybus_phasecut!(net::Dict;sigma_noise=0.1)
    PM.compute_ac_pf!(net)
    n_bus = length(net["bus"])

    #Problem params
    Y = Matrix(calc_basic_admittance_matrix(net))
    Vrect = PM.calc_basic_bus_voltage(net)
    S = PM.calc_basic_bus_injection(net)
    Imag = abs.(S) ./ abs.(Vrect)

    Irect = Y*Vrect
    Iangle = angle.(Irect)
    Imag_true = abs.(Irect)
    @assert Diagonal(Imag_true)*cis.(Iangle) ≈ Irect ≈ Diagonal(Imag)*cis.(Iangle)

    #Bus type idx
    slack_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[3])

    model = Model(SCS.Optimizer)
    
    #set_optimizer_attribute(model, "max_iter", 6000)
    #Complex voltage
    @variable(model,vr[1:n_bus])
    @variable(model,vi[1:n_bus])

    #Complex current phase
    @variable(model,ur[1:n_bus])
    @variable(model,ui[1:n_bus])

    #PSD matrix
    @variable(model,X[1:2*n_bus,1:2*n_bus], PSD)

    #Construct M matrix (fixed phase representation)
    M = Diagonal(Imag)*(I - Y * pinv(Y))*Diagonal(Imag)
    #transformed M matrix
    Mr,Mi = real.(M),imag.(M)
    @expression(model,T_M,
        [
            Mr -1*Mi; 
            Mi Mr
        ]
    )

    #PQ/PV BUS Constraints
    for k=1:n_bus
        if k ∈ slack_idx #|| k ∈ pv_idx
            for j =1:n_bus
                u_k = cis(Iangle[k]) #cos(Iangle[k]) + sin(Iangle[k])*im
                u_j = cis(Iangle[j]) #cos(Iangle[j]) + sin(Iangle[j])*im

                Ukj = u_k*conj(transpose(u_j))
                Real_Ukj = real(Ukj)
                Imag_Ukj = imag(Ukj)

                @constraint(model,X[k,j]== Real_Ukj)
                @constraint(model,X[k+n_bus,j] == Imag_Ukj)
                @constraint(model,X[k,j+n_bus] == -Imag_Ukj)
                @constraint(model,X[k+n_bus,j+n_bus] == Real_Ukj)
            end
            
        end
    end

    #Complex constraints
    for i = 1:n_bus
        @constraint(model,X[i,i] ==1)
        @constraint(model,X[i+n_bus,i+n_bus]==1)
    end

    #Trace objective
    @objective(
        model,
        Min,
        tr(T_M*X)
    )
    optimize!(model)

    #Extract solution, check for uncertainty of the relaxation
    X_opt = value.(X)[1:n_bus,1:n_bus] + value.(X[1+n_bus:end,1:n_bus]) .*im
    Xr1 = PPR.calc_closest_rank_r(X_opt,1)
    u = Xr1[:,1]
    Iangle_est = angle.(u)
end