"""
Ybus phase retrieval problem data
"""
struct YbusPhretData
    case_name::String
    sigma_noise::Float64 #standard deviation of the noise of the current mag measurement
    n_bus::Int
    slack_idx::Union{AbstractArray,Set}
    pv_idx::Union{AbstractArray,Set}
    pq_idx::Union{AbstractArray,Set}
    Y::AbstractMatrix
    Vrect::AbstractArray{Complex}
    Vmag::AbstractArray{Real} #voltage magnitudes
    Vangle::AbstractArray{Real} #voltage angles
    s::AbstractArray{Complex} #Rectangular complex power
    p::AbstractArray{Real} #Real power
    q::AbstractArray{Real} #Reactive power
    Irect::AbstractArray{Complex}  #Rectangular current phasor ground truth Y*v_rect_groundtruth
    Imag::AbstractArray{Real} #Current magnitude
    Iangle::AbstractArray{Real} #Current angles
    Imag_obs::AbstractArray{Real} #Observed noisy current magnitudes.
end

"""
Ybus phase retrieval solution data
"""
struct YbusPhretSolution
    model::Model
    data::YbusPhretData
    X_opt::AbstractArray #symmetric matrix solution
    Vrect::AbstractArray{Complex}
    Irect::AbstractArray{Complex} #Rectangular estimated current phasors
    Vangle::AbstractArray{Real} #Voltage phase
    Iangle::AbstractArray{Real} #Current phase
    Vrect_rel_err::Real #Relative percentage error ||v_rect_est - Y^{-1} Irect_gtruth||_2/||Y^{-1} Irect_gruth|| 
    Irect_rel_err::Real #Relative perctange error  ||Y v_rect_gtruth - Irect_est||_2
    Vangle_rel_err::Real 
    Iangle_rel_err::Real
    Vangle_sq_err::AbstractArray{Float64}
    Iangle_sq_err::AbstractArray{Float64}
end

"""
Given a basic network data dict make the ybus phret PROBLMEM DATA
"""
function YbusPhretData(net::Dict{String,Any};sigma_noise=0,mean_noise=0)
    #---case name
    case_name = net["name"]

    #Slack buses
    slack_idx = calc_bus_idx_of_type(net,[3])
    pq_bus_idx = calc_bus_idx_of_type(net,[1])
    pv_bus_idx = calc_bus_idx_of_type(net,[2])
    n_bus = length(net["bus"])

    #topology params
    Y = Matrix(calc_basic_admittance_matrix(net)) #full admittance matrix

    #voltages
    Vrect = calc_basic_bus_voltage(net)
    Vmag,Vangle = abs.(Vrect),angle.(Vrect)

    #----- powers
    s = calc_basic_bus_injection(net)
    p,q = real.(s),imag.(s)

    #----- currents
    Irect = Y*Vrect
    Iangle = angle.(Irect)
    Imag = abs.(Irect)
    
    #--- Compute noisy observations of the current magnitude
    noise_dist = [] #noise_dist = [Normal(mean_noise,sigma_noise*Imag_i) for Imag_i in Imag] #noise distribution
    for (i,Imag_i) in enumerate(Imag)
        if i ∈ pq_bus_idx 
            push!(noise_dist,Normal(mean_noise,sigma_noise*Imag_i))
        else
            push!(noise_dist,Normal(mean_noise,0))
        end
    end
    n = [rand(d_i) for d_i in noise_dist]
    Imag_obs = abs.(Y*Vrect) .+ n #(sqrt.(p.^2 + q.^2) ./ Vmag)

    #--- Sanity checks
    #@assert all(abs.(s)./Vmag .≈ Imag)
    @assert all([n_i ≈ 0.0 for (i,n_i) in enumerate(n) if i ∉ pq_bus_idx]) 
    @assert Diagonal(Imag)*cis.(Iangle) ≈ Irect ≈ Diagonal(Imag)*cis.(Iangle)
    @assert norm(abs.(Irect) - Imag) <= 1e-3 "Ybus multiplication check failed with "*string(norm(abs.(Irect) - Imag))

    return YbusPhretData(
        case_name,
        sigma_noise,
        n_bus,
        slack_idx,
        pv_bus_idx,
        pq_bus_idx,
        Y,
        Vrect,
        Vmag,
        Vangle,
        s,
        p,
        q,
        Irect,
        Imag,
        Iangle,
        Imag_obs
        )
end


"""
Given a basic network data dict 
make the ybus phret data, solve the AC PF, and solve the Ybus PhaseCut algorithm.
"""
function solve_ybus_phasecut!(net::Dict{String,Any})
    compute_ac_pf!(net)
    data = YbusPhretData(net)
    sol = solve_ybus_phasecut(data)
    return sol
end

# """
# Given a basic network data dict and a measurement nois level,
# make the ybus phret data , solve the AC PF, and solve the Ybus PhaseCut algorithm.
# """
# function solve_ybus_phasecut!(net::Dict{String,Any},sigma_noise::Float64)
#     compute_ac_pf!(net)
#     data = YbusPhretData(net,sigma_noise=sigma_noise)
#     sol = solve_ybus_phasecut(data)
#     return sol
# end

"""
Given a basic network data dict and an array of measurement noise,
make the ybus phret data , solve the AC PF, and solve the Ybus PhaseCut algorithm.

Returns Array{YbusPhretSolution}
"""
function solve_ybus_phasecut!(net::Dict{String,Any},sigma_noise::Array{Float64})
    solutions = []
    for s in sigma_noise
        compute_ac_pf!(net)
        data = YbusPhretData(net,sigma_noise=s)
        sol = solve_ybus_phasecut(data)
        push!(solutions,sol)
    end
    return solutions
end


"""
Given a basic network data dict make the ybus phret MODEL
"""
function solve_ybus_phasecut(data::YbusPhretData)
    #unpack data
    n_bus,slack_idx,pq_idx,pv_idx = data.n_bus,data.slack_idx,data.pq_idx,data.pv_idx
    Y,Vmag,Imag = data.Y,data.Vmag,data.Imag 
    Irect,Vrect = data.Irect,data.Vrect
    Iangle,Vangle = data.Iangle, data.Vangle
    Imag_obs = data.Imag_obs #--- Noisy observations

    #----- Start model
    model = Model(SCS.Optimizer)
    
    #Complex voltage
    @variable(model,vr[1:n_bus])
    @variable(model,vi[1:n_bus])
    
    #Complex current phase
    @variable(model,ur[1:n_bus])
    @variable(model,ui[1:n_bus])

    #PSD matrix
    @variable(model,X[1:2*n_bus,1:2*n_bus], PSD)

    #Construct M matrix (fixed phase representation)
    M = Diagonal(Imag_obs)*(I - Y * pinv(Y))*Diagonal(Imag_obs)
    #transformed M matrix
    Mr,Mi = real.(M),imag.(M)
    @expression(model,T_M,
        [
            Mr -1*Mi; 
            Mi Mr
        ]
    )

    #Slack bus Constraints
    for k=1:n_bus
        if k ∈ slack_idx || k ∈ pv_idx
            for j =1:n_bus
                u_k = cis(Iangle[k]) #cos(Iangle[k]) + sin(Iangle[k])*im
                u_j = cis(Iangle[j]) #cos(Iangle[j]) + sin(Iangle[j])*im

                Ukj = u_k*conj(transpose(u_j))
                re_Ukj = real(Ukj)
                im_Ukj = imag(Ukj)

                @constraint(model,X[k,j]== re_Ukj)
                @constraint(model,X[k+n_bus,j] == im_Ukj)
                @constraint(model,X[k,j+n_bus] == -im_Ukj)
                @constraint(model,X[k+n_bus,j+n_bus] == re_Ukj)
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
    X_opt = value.(X)[1:n_bus,1:n_bus] + value.(X)[1+n_bus:end,1:n_bus] .*im
    X_opt = calc_closest_rank_r(X_opt,1)
    u = X_opt[:,1]
    u = u ./ abs.(u)
    @assert all([abs(u_i) ≈ 1 for u_i in u])

    #reconstruct the current phase, voltage phase, etc.
    Iangle_est = angle.(u)
    Iangle_est[slack_idx] = Iangle[slack_idx]
    Iangle_est[pv_idx] = Iangle[pv_idx]
    #Irect_est = Imag_obs .* cis.(Iangle_est)
    Irect_est = Diagonal(Imag_obs)*u
    Irect_est[slack_idx] = Irect[slack_idx]
    Irect_est[pv_idx] = Irect[pv_idx]
    #Irect_est = Imag_obs .* (cos.(Iangle_est) + sin.(Iangle_est)*im)
    Vrect_est = inv(Y)*Irect_est
    Vangle_est = angle.(Vrect_est)

    #Return the solution struct
    return YbusPhretSolution(
        model, #model struct
        data, #data struct
        X_opt,
        Vrect_est,
        Irect_est,
        Vangle_est,
        Iangle_est,
        norm(Vrect_est - data.Vrect)/norm(data.Vrect)*100,#Vrect_rel_err,
        norm(Irect_est - data.Irect)/norm(data.Irect)*100,#Irect_rel_err,
        norm(Vangle_est - data.Vangle)/norm(data.Vangle)*100,#Vangle_rel_err,
        norm(Iangle_est - data.Iangle)/norm(data.Iangle)*100,#Iangle_rel_er
        abs.(Vangle_est .- data.Vangle).^2, #squared errors
        abs.(Iangle_est .- data.Iangle).^2 #squared errors
    )
end



#----- Deprecated: PQ bus ybus phasecat


function solve_pq_ybus_phasecut(data::YbusPhretData,net::Dict{String,Any})
    n_bus,Y,Vmag,Imag = data.n_bus, data.Y, data.Vmag, data.Imag #unpack data
    Iangle,Vangle = data.Iangle, data.Vangle
    

    
    #Bus type idx
    pq_idx = calc_bus_idx_of_type(net,[1])
    pv_idx = calc_bus_idx_of_type(net,[2])
    pq_pv_idx = sort(pq_idx ∪ pv_idx)
    slack_idx = calc_bus_idx_of_type(net,[3])

    #transformed Y bus
    Yr,Yi = real.(Y),imag.(Y)
    T_Y = [Yr -1*Yi;
            Yi Yr]
    

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 6000)
    

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

    #Complex current
    @variable(model,I_u_real[1:n_bus])
    @variable(model,I_u_imag[1:n_bus])
    for i =1:n_bus 
        # @constraint(model,I_u_real[i]<= 1)
        # @constraint(model,I_u_imag[i]<= 1)
        # @constraint(model,-I_u_real[i]<=0)
        # @constraint(model,-I_u_imag[i]<=0)
        @constraint(model,I_u_real[i]^2 + I_u_imag[i]^2 ==1) #Unit circle phase angles
        if i ∈ pv_idx || i ∈ slack_idx #PV bus current phase angles are known
            @constraint(model,I_u_real[i] == Imag[i]*cos(Iangle[i]) + 1e-9)
            @constraint(model,I_u_imag[i] == Imag[i]*sin(Iangle[i]) + 1e-9)
        end
    end
    #Transformed current magnitude

    @expression(model,Iphasor,
    [
        Diagonal(Imag)*I_u_real;
        Diagonal(Imag)*I_u_imag
    ]
    )

    # #Complex voltage 
    # @variable(model,V_u_real[1:n_bus])
    # @variable(model,V_u_imag[1:n_bus])
    # for i =1:n_bus
    #     @constraint(model,V_u_real[i]^2 + V_u_imag[i]^2 ==1) #Unit circle phase angles
    #     if i ∈ pv_idx || i ∈ slack_idx #PV bus phase angles are known
    #         @constraint(model,V_u_real[i] == Vmag[i]*cos(Vangle[i]))
    #         @constraint(model,V_u_imag[i] == Vmag[i]*sin(Vangle[i]))
    #     end
    # end
    # @expression(model,Vphasor,
    #     [
    #         Diagonal(Vmag)*V_u_real;
    #         Diagonal(Vmag)*V_u_imag    
    #     ]
    # )
  
    #@expression(model,Vphasor,
    #     pinv(T_Y)*[Diagonal(Imag) zeros(n_bus,n_bus);zeros(n_bus,n_bus) Diagonal(Imag)] *[I_u_real ; I_u_imag]
    # )
       
    
    
    #Residual expression
    # @variable(model,residuals)
    # @constraint(model,residuals .==  Iphasor .- T_Y*Vphasor    
    #     #Iphasor[pq_idx] .- (T_Y*[uv_r;uv_im])[pq_idx] #TRY: PQ bus only
    #     #Iphasor[Union(pq_idx,pv_idx)] .- (T_Y*[uv_r;uv_im])[Union(pq_idx,pv_idx)] #TRY: PQ and PV bus only
    # )
    
    #Objective
    @objective(model,Min,
        sum([I_u_real_k*Q_k for (I_u_real_k,Q_k) in zip(I_u_real[pq_idx], (Mr*I_u_real .- Mi*I_u_imag)[pq_idx] )])  
        -sum([I_u_imag_k*Q_k for (I_u_imag_k,Q_k) in zip(I_u_imag[pq_idx], (Mi*I_u_real .+ Mr*I_u_imag)[pq_idx] )])
        #sum(residuals.^2)
    )
    optimize!(model)
    return value.(I_u_real) .+ value.(I_u_imag).*im
end