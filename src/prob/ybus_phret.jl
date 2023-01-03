"""
Ybus phase retrieval problem data
"""
struct YbusPhretData
    n_bus::Int
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
end


"""
Given a basic network data dict make the ybus phret PROBLMEM DATA
"""
function YbusPhretData(net::Dict{String,Any};sel_bus_types=[1,2])
    compute_ac_pf!(net)
    
    #Select buses
    sel_bus_idx = calc_bus_idx_of_type(net,sel_bus_types)
    n_sel_bus = length(sel_bus_idx) #Get num_bus
    
    n_bus = length(net["bus"])

    #topology params
    Y = Matrix(calc_basic_admittance_matrix(net)) #full admittance matrix
    Yprime = Matrix(calc_basic_admittance_matrix(net))[sel_bus_idx,sel_bus_idx] #admittance matrix with slack removed

    #voltages
    Vrect = calc_basic_bus_voltage(net)
    Vmag,Vangle = abs.(Vrect),angle.(Vrect)

    #powers
    s = calc_basic_bus_injection(net)
    p,q = real.(s),imag.(s)
    
    #currents
    Irect = Y*Vrect
    Iangle = angle.(Irect)
    Imag =  abs.(s) ./ abs.(Vrect)
    
    # println("Vmag",Vmag)
    # println("Irect ",Irect)
    # println("Iangle ",Iangle)
    # println("Imag",Imag)

    @assert norm(abs.(Irect) - Imag) <= 1e-3 "Ybus multiplication check failed with "*string(norm(abs.(Irect) - Imag))

    return YbusPhretData(n_bus,Y,Vrect,Vmag,Vangle,s,p,q,Irect,Imag,Iangle)
end


"""
Given a basic network data dict make the ybus phret MODEL
"""
function solve_ybus_phasecut(data::YbusPhretData,net::Dict{String,Any})
    n_bus,Y,Vmag,Imag = data.n_bus,data.Y,data.Vmag,data.Imag #unpack data
    Iangle,Vangle = data.Iangle, data.Vangle


    #Bus type idx
    pq_idx = calc_bus_idx_of_type(net,[1])
    pv_idx = calc_bus_idx_of_type(net,[2])
    pq_pv_idx = sort(pq_idx ∪ pv_idx)
    slack_idx = calc_bus_idx_of_type(net,[3])


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
        if k ∈ pv_idx || k ∈ slack_idx
            for j =1:n_bus
                u_k = cos(Iangle[k]) + sin(Iangle[k])*im
                u_j = cos(Iangle[j]) + sin(Iangle[j])*im

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
        #@constraint(model,X[1:n_bus,i] .== ur*ur[i])
        #@constraint(model,X[n_bus+1:end,i] .== ui*ur[i])
        #@constraint(model,X[1:n_bus,i+n_bus] .== -ur.*ui[i])
        #@constraint(model,X[n_bus+1:end,i+n_bus] .== ui*ui[i])
    end

   #Trace objective
    @objective(
        model,
        Min,
        tr(T_M*X)
    )
    optimize!(model)
   
    #Extract solution, check for uncertainty of the relaxation
    #X_opt = value.(X)[1:n_bus,1:n_bus] + value.(X[1+n_bus:end,1:n_bus]) .*im
    X_opt = value.(X)
    # evals = eigvals(X_opt)
    # if length([e for e in evals if abs(real(e)) > 1e-4]) >1
    #     X_hat = calc_closest_rank_r(X_opt,1)
    # end
    #X_hat = X_opt[1:n_bus,1:n_bus] .+ X_opt[n_bus+1:end,1:n_bus] .* im
    evals,evecs = eigen(X_opt)
    uhat = evecs[:,1] ./ abs.(evecs[:,1])
    
    #reconstruct the current phase, voltage phase, etc.
    Iangle_est = angle.(uhat)
    Irect_est = Diagonal(Imag)*uhat
    #Irect_est = Imag .* (cos.(Iangle_est) + sin.(Iangle_est)*im)
    Vrect_est = pinv(Y)*Diagonal(Imag)*uhat
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
        norm(Iangle_est - data.Iangle)/norm(data.Iangle)*100#Iangle_rel_er
    )
end

"""
Given a basic network data dict make the ybus phret data 
and solve with the PhaseCut algorithm.
"""
function solve_ybus_phasecut!(net::Dict{String,Any})
    data = YbusPhretData(net)
    sol = solve_ybus_phasecut(data,net)
    return sol
end



#Vangle_rect = 
# println("Rank of solution: ",rank(X_opt))
# is_tight,uncertainty = true,nothing
# if rank(X_opt) > 1
#     is_tight = false
#     X_rank_one = calc_closest_rank_r(X_opt,1)
#     #Compute uncertainty of the relaxation
#     vecs = eigvecs(X_rank_one)
#     v = vecs[:,1] ./ norm(vecs[:,1])
#     uncertainty = diag(X_opt - v*v')
#     #Force rank one to the best possible solution
#     X_opt = X_rank_one #save the 
# end


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


# =====
# Greedy algorithm
# =====

struct GreedyYbusPhretSolution
    data::YbusPhretData
    u_hat::Vector
    th_hat::Vector
    irect_hat::Vector
    rel_err::Float64
    u_iter::Vector{Vector{ComplexF64}} #iterations of the greedy algorithm
end
"""
Solve the greedy ybus phasecut problem from "Phase Recovery, Maxcut and Complex Semidefinite Programming"
"""
function solve_greedy_ybus_phasecut(data::YbusPhretData;sigma_init_phase=0.1,max_iter=5,ϵ=1e-9)
    n_bus,Y,Vmag,Imag = data.n_bus,data.Y,data.Vmag,data.Imag #unpack data
    M = Diagonal(Imag)*(I - Y*pinv(Y))*Diagonal(Imag)
    #initialize the phase
    dist_uph = π/2*Normal(0,sigma_init_phase)
    init_Th = rand(dist_uph,n_bus)
    u = [Imag_i*cos(θ_i) + Imag_i*sin(θ_i)*im for (Imag_i,θ_i) in zip(Imag,init_Th)]
    u = [u_i/abs(u_i) for u_i in u]
    #u_iter = zeros(max_iter,length(u))
    #u_iter[1,:] = u
    #while (k>1 && norm(u_iter[k]-norm(u_iter[k-1]))>ϵ) && k<=max_iter
    #k=1
    for k=1:max_iter
        for i=1:n_bus 
            off_diag_sum = sum([M[j,i]*u[j] for j=1:n_bus if j !== i])
            if Imag[i] < 1e-3 || norm(u[i])<1e-6
                u[i] = 1e-6 + 1e-6*im
                continue
            end
            u[i] = (-off_diag_sum)/(abs(off_diag_sum))
        end
        #k += 1
    end
    return u
    #u_hat = u_iter[end]
   # @assert all([abs(u_i)==1 for u_i in u_iter[end]])
    # return GreedyYbusPhretSolution(
    #     data,
    #     u_iter[end],
    #     angle.(u_iter[end]),
    #     Diagonal(Imag)*u_hat,
    #     norm(angle.(u_hat)-data.Iangle)/norm(data.Iangle)*100,
    #     u_iter
    # )
end

function solve_greedy_ybus_phasecut!(net::Dict{String,Any};sigma_init_phase=0.1,max_iter=1000,ϵ=1e-9)
    data = YbusPhretData(net)
    u_iter = solve_greedy_ybus_phasecut(data,sigma_init_phase=sigma_init_phase,max_iter=max_iter,ϵ=ϵ)
    return u_iter
end