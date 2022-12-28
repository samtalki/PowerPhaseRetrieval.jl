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
    is_tight::Bool 
    uncertainty::Union{AbstractArray,Nothing} #measure of uncertainty diag(U - v*v^T) around the solution described in "Phase Recovery, Maxcut, ...."
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
function YbusPhretData(net::Dict{String,Any})
    compute_ac_pf!(net)
    
    #topology param
    n_bus = length(net["bus"])
    Y = Matrix(calc_basic_admittance_matrix(net))
    
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
function solve_ybus_phasecut(data::YbusPhretData)
    n_bus,Y,Vmag,Imag = data.n_bus,data.Y,data.Vmag,data.Imag #unpack data
    model = Model(SCS.Optimizer)
    

    #Construct M matrix (fixed phase representation)
    M = Diagonal(Imag)*(I - Y * Y')*Diagonal(Imag)
    Mr,Mi = real.(M),imag.(M)

    #transformed M matrix
    @expression(model,T_M,
        [
            Mr -1*Mi; 
            Mi Mr
        ]
    )

    #Phase matrix X = uu*
    @variable(model,X[1:2*n_bus,1:2*n_bus],PSD)
    
    #Phase matrix constraints
    #for i=1:n_bus
        # @constraint(model,X[i,i] + X[n_bus+i,n_bus+i] == 2)
        # for j=1:n_bus
        #     @constraint(model,X[i,j] == X[n_bus+i,n_bus+j])
        #     @constraint(model,X[n_bus+i,j] == -X[i,n_bus+j])
        # end   
    #end
    
    for i =1:n_bus*2
        @constraint(model,X[i,i]==1) 
    end

    #objective and solve
    @objective(model,Min,tr(T_M*X))
    optimize!(model)

   
    #Extract solution, check for uncertainty of the relaxation
    X_opt = value.(X)[1:n_bus,1:n_bus] + value.(X)[n_bus+1:end,1:n_bus].* im
    (values,vectors) = eigen(X_opt)
    u = vectors[:,1] ./ abs.(vectors[:,1])
    uncertainty = diag(X_opt - u*u')
    
    #reconstruct the current phase, voltage phase, etc.
    Iangle_est = angle.(u)
    Irect_est = Imag .* (cos.(Iangle_est) + sin.(Iangle_est)*im)
    Vrect_est = pinv(Y)*Diagonal(Imag)*u
    #Vrect_est = inv(Y)*Irect_est
    Vangle_est = angle.(Vrect_est)

    Vangle_rect = 
    println("Rank of solution: ",rank(X_opt))
    is_tight,uncertainty = true,nothing
    if rank(X_opt) > 1
        is_tight = false
        X_rank_one = calc_closest_rank_r(X_opt,1)
        #Compute uncertainty of the relaxation
        vecs = eigvecs(X_rank_one)
        v = vecs[:,1] ./ norm(vecs[:,1])
        uncertainty = diag(X_opt - v*v')
        #Force rank one to the best possible solution
        X_opt = X_rank_one #save the 
    end


    #Return the solution struct
    return X_opt,YbusPhretSolution(
        model, #model struct
        data, #data struct
        is_tight,
        uncertainty,
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
    sol = solve_ybus_phasecut(data)
    return sol
end