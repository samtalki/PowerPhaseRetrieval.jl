using JuMP 
using Ipopt
using LinearAlgebra
using PowerModels
using Distributions
using Random



"""
Solve the rectangular phase retrieval problem given a power models network.

Parameters: 
    net::Dict{String,Any} - a power models network
    μ::Float64 - mean of the noise
    σ::Float64 - standard deviation of the noise
"""
function solve_rectangular_phret!(net::Dict{String,Any};μ=0.0,σ=0.01)
    # run power flow
    compute_ac_pf!(net)

    # process data
    pq_buses = calc_bus_idx_of_type(net,[1])
    v0 = calc_basic_bus_voltage(net)[pq_buses]
    Y = calc_basic_admittance_matrix(net)[pq_buses,pq_buses]
    e0,f0 = real.(v0),imag.(v0) # real and imaginary part of the voltage
    G,B = real.(Y),imag.(Y) # real and imaginary admittance

    # ---- get the sensitivities
    ∂pe,∂qe,∂pf,∂qf,∂v²e,∂v²f = calc_rectangular_jacobian_sensitivities(G,B,v0)
    J = [
        ∂pe  ∂pf;
        ∂qe  ∂qf;
        ∂v²e  ∂v²f
    ]

    # ---- get the dimensionality of the problem
    n = length(pq_buses) # number of buses
    n3,n2 = size(J)

    # ---- get the true measurements
    x_true = [e0;f0]
    f_x_true = J*x_true

    # ---- get the observed measurements
    f_x_obs = abs.(f_x_true) + rand(Normal(μ,σ),n3)
    p_obs,q_obs,vm²_obs = f_x_obs[1:n],f_x_obs[n+1:2*n],f_x_obs[2*n+1:end]
    vm_obs = sqrt.(vm²_obs)

    # ------ Create and solve the model
    model = build_rectangular_phret_model(vm_obs,p_obs,q_obs,∂pe,∂pf,∂qe,∂qf,∂v²e,∂v²f)

    # Optimize the model
    optimize!(model)

    # Get the solution
    u_hat = value.(model[:u])
    v_hat = value.(model[:vhat])
    x_hat = [real.(v_hat); imag.(v_hat)]

    # ---- compute the error
    err = norm(x_hat-x_true)/norm(x_true)

    # ---- return the solution
    return Dict(
        "x_true" => x_true,
        "x_hat" => x_hat,
        "err" => err
    )
end


"""
Builds a rectangular phase retrieval model given a nominal operating point and the admittance matrix. 
Uses the sensitivities of the power-rectangular complex voltage equations to build the model.

Parameters:
    vm_obs::Vector{Float64} -- observed voltage magnitude
    p_obs::Vector{Float64} -- observed active power
    q_obs::Vector{Float64} -- observed reactive power
    Y::AbstractMatrix{ComplexF64} -- admittance matrix

"""
function build_rectangular_phret_model(vm_obs,p_obs,q_obs,∂pe,∂pf,∂qe,∂qf,∂v²e,∂v²f)
    # process data
    n = length(vm_obs) # number of buses
   
    # Create the model
    model = Model(Ipopt.Optimizer)

    # Define the variables
    @variable(model, u[1:n] in ComplexPlane()) # complex voltage phase --- normalized to be one
    @variable(model, vhat[1:n] in ComplexPlane()) # complex voltage phase --- normalized to be vm_obs in magnitude
    
    # expression -- complex voltage phase scaled by the known voltage magnitude
    xhat = [
        real(vhat);
        imag(vhat)
    ]

    # --- Define the Jacobian and observed mismatches
    Jx = [
        ∂pe ∂pf;
        ∂qe ∂qf;
        ∂v²e ∂v²f
    ]
    fx_obs = [
        p_obs;
        q_obs;
        vm_obs.^2
    ]

    # --- Define the constraints
    for i = 1:n
        @constraint(model, real(u[i])^2 + imag(u[i])^2 == 1) #--- normalize the voltage phase
        @constraint(model, vhat[i] == vm_obs[i]*u[i]) #--- normalize the voltage phasor
    end

    # --- Define the loss
    fx_hat = Jx*xhat #--- predicted mismatches
    # --- Define the residuals
    @variable(model, residuals[1:3*n])
    @constraint(model, [i=1:3*n], residuals[i] == fx_hat[i] - fx_obs[i])
    loss = sum([residual_i^2 for residual_i in residuals]) #--- loss

    # --- Define the objective
    @objective(model, Min, loss)
   
    return model
end



"""
Builds a rectangular phase retrieval model given a nominal operating point and the admittance matrix. 
Uses the sensitivities of the power-rectangular complex voltage equations to build the model.

Parameters:
    vm_obs::Vector{Float64} -- observed voltage magnitude
    p_obs::Vector{Float64} -- observed active power
    q_obs::Vector{Float64} -- observed reactive power
    Y::AbstractMatrix{ComplexF64} -- admittance matrix

"""
function build_rectangular_phret_model(vm_obs,p_obs,q_obs,Y::AbstractMatrix{ComplexF64})
    # process data
    n = length(vm_obs) # number of buses
    G,B = real.(Y),imag.(Y) # real and imaginary admittance

    # Create the model
    model = Model(Ipopt.Optimizer)

    # Define the variables
    @variable(model, u[1:n] in ComplexPlane()) # complex voltage phase --- normalized to be one
    @variable(model,∂pe[1:n,1:n]) # ∂p/∂e
    @variable(model,∂qe[1:n,1:n]) # ∂q/∂e
    @variable(model,∂pf[1:n,1:n]) # ∂p/∂f
    @variable(model,∂qf[1:n,1:n]) # ∂q/∂f
    @variable(model,∂v²e[1:n,1:n]) # ∂v^2/∂e
    @variable(model,∂v²f[1:n,1:n]) # ∂v^2/∂f

    # expression -- complex voltage phase scaled by the known voltage magnitude
    vhat = vm_obs.*u

    # Define the constraints
    for i = 1:n
        for k=1:n
            if k != i # off-diagonal elements
                ∂pe[i,k] == G[i,k]*real(vhat[i]) + B[i,k]*imag(vhat[i])
                ∂pf[i,k] == G[i,k]*imag(vhat[i]) - B[i,k]*real(vhat[i])
                ∂qe[i,k] == ∂pf[i,k]
                ∂qf[i,k] == -∂pe[i,k]
                ∂v²e[i,k] == 0
                ∂v²f[i,k] == 0

            else # diagonal elements

                #--- active power to  complex voltage
                ∂pe[i,i] == sum(
                    [G[i,k]*real(vhat[k])-B[i,k]*imag(vhat[k]) for k=1:n]
                ) + G[i,i]*real(vhat[i]) + B[i,i]*imag(vhat[i])
                ∂pf[i,i] == sum(
                    [G[i,k]*imag(vhat[k])+B[i,k]*real(vhat[k]) for k=1:n]
                ) + G[i,i]*imag(vhat[i]) - B[i,i]*real(vhat[i])

                #--- reactive power to complex voltage
                ∂qe[i,i] == sum(
                    [-G[i,k]*imag(vhat[k]) - B[i,k]*real(vhat[k]) for k=1:n]
                ) - B[i,i]*real(vhat[i]) + G[i,i]*imag(vhat[i])
                ∂qf[i,i] == sum(
                    [G[i,k]*real(vhat[k]) - B[i,k]*imag(vhat[k]) for k=1:n]
                ) - G[i,i]*real(vhat[i]) - B[i,i]*imag(vhat[i])

                #--- squared voltage to complex voltage
                ∂v²e[i,i] == 2*real(vhat[i])
                ∂v²f[i,i] == 2*imag(vhat[i])
            end
        end
    end


    # --- Define the Jacobian and observed mismatches
    Jx = [
        ∂pe ∂pf;
        ∂qe ∂qf;
        ∂v²e ∂v²f
    ]
    fx_obs = [
        p_obs;
        q_obs;
        vm_obs.^2
    ]

    # --- Define the loss
    fx_hat = Jx*[real(vhat);imag(vhat)] #--- predicted mismatches
    # --- Define the residuals
    @variable(model, residuals[1:3*n])
    @constraint(model, [i=1:3*n], residuals[i] == fx_hat[i] - fx_obs[i])
    loss = sum([residual_i^2 for residual_i in residuals]) #--- loss

    # --- Define the objective
    @objective(model, Min, loss)

    # --- Define the constraints
    for i = 1:n
        @constraint(model, real(u[i])^2 + imag(u[i])^2 == 1) #--- normalize the voltage phase
    end

    return model
end