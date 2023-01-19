using JuMP,LinearAlgebra,Gurobi
struct CurrentPhaseMax
    Y::AbstractMatrix #admittance matrix
    y0::Vector{ComplexF64} #ybus anchor vector
    ℓmag::Vector{Float64} #current magnitudes
    vmag::Vector{Float64} #voltage magnitudes
    ℓrect::Vector{ComplexF64} #ground truth current phasor
    vrect::Vector{ComplexF64} #gruth voltage phasor
    anchor_satisfied::Bool #Whether or not the anchor vector condition from Romberg is satisfied
end


"""
Given net dict, make phasemax model
"""
function CurrentPhaseMax(net::Dict{String,Any})
    compute_ac_pf!(net)

    vrect = calc_basic_bus_voltage(net)
    vmag = abs.(vrect)
    s = calc_basic_bus_injection(net)
    ℓmag = abs.(s) ./ vmag

    #compute admittance anchor
    Y = calc_basic_admittance_matrix(net)
    ℓrect = Y*vrect
    (n,n) = size(Y)
    Σ = (1/n)*ones(size(Y))
    for (ℓmag_i,y_i) in zip(ℓmag,eachrow(Y))
        Σ += ℓmag_i*(y_i * conj(y_i'))
    end
    y0 = eigvecs(Σ)[:,1]
    anchor_satisfied,δ = calc_romberg_bound(y0,vrect)

    return CurrentPhaseMax(
        Y,y0,ℓmag,vmag,ℓrect,vrect,anchor_satisfied
    )
end

"""
Given an anchor admittance matrix vector y0 and the phasor voltages, find the 
δ that satisfies rombergs bound
"""
function calc_romberg_bound(y0,vrect;ϵ=1e-3)
    satisfied = false
    δ_star  = nothing
    for δ = ϵ:1e-3:1-ϵ
        if abs(dot(y0,vrect)) >= δ*norm(y0)*norm(vrect)
            satisfied = true
            δ_star = δ
            println("statisfied w δ=",δ_star)
            break
        end    
    end
    return satisfied,δ_star
end

function solve_ybus_phasemax!(net::Dict)
    data = CurrentPhaseMax(net)
    y0_re,y0_im = real.(data.y0),imag.(data.y0)
    model = Model(Gurobi.Optimizer)
    @variable(model,vreal[1:n])
    @variable(model,vimag[1:n])
    @constraint(model)
end