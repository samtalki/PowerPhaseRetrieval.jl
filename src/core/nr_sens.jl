# Newton-raphson power flow with automatic differentiation.

using LinearAlgebra, SparseArrays
import ForwardDiff
import PowerModels as PM

abstract type SensitivityModel end

"""
Type that comprises a standard Newton-Raphson sensitivity model for a power network
    data:: PowerModels network data
    Y:: Admittance matrix 
    J:: A function J: x -> [ ∂p/∂θ  ∂q/∂θ ; ∂p/∂v ∂q/∂v](x) that evaluates the power flow jacobian given a vector of voltage phasors.
"""
struct LinearSensitivityModel <: SensitivityModel 
    data::Dict{String, Any} #PowerModels network
    Y::SparseMatrixCSC #Network admittance matrix
    J::Function # J: vph ∈ R²ⁿ ↦ Δx ∈ R²ⁿ. A function that returns the power flow Jacobian through automatic differentiatiation
    f::Function # f: vph ∈ R^2n ↦ [Δp,Δq]^T ∈ R^2n. Mistaches given a  grid operating point x.
    f_approx::Function # f: vph ↦ [p,q]^T ∈ R^2n
end

"""
Constructor for the basic network sensitivty model.
    data:: PowerModels network
"""
function LinearSensitivityModel(data::Dict{String, Any})
    s_inj = PM.calc_basic_bus_injection(data)
    x0 = PM.calc_basic_bus_voltage(data)
    Y = PM.calc_basic_admittance_matrix(data)
    f = x::AbstractArray -> calc_mismatch(x,s_inj,Y)
    J = x::AbstractArray -> -1 .* ForwardDiff.jacobian(f,x) #Note: x = [θ ; vmag]
    f_approx =  x::AbstractArray -> s_inj + J(x0)*(x0-x)
    return LinearSensitivityModel(data,Y,J,f,f_approx)
end

"""
Given complex voltage and powers, and an admittance matrix Y, calculate the power flow mismatch.
"""
function calc_mismatch(x,s,Y)
    n_bus = size(Y,1)
    # Convert phasor to rectangular
    function calc_rect_bus_voltage(x::AbstractArray)
        θ,vm = x[1:n_bus],x[n_bus+1:end]
        [vmᵢ*(cos(θᵢ) + sin(θᵢ)*im) for (θᵢ,vmᵢ) in zip(θ,vm)]
    end
    v_rect = calc_rect_bus_voltage(x)
    si = v_rect .* conj(Y * v_rect) #compute the injection
    Δp, Δq = real(s - si), imag(s - si)
    Δx = [Δp ; Δq]
    return Δx
end

"""
1st definition: 
given a PowerModels network, returns a [vangle; vmag] vector of bus voltage phasor quantities
as they appear in the network data.
"""
function calc_phasor_bus_voltage(data::Dict{String, Any})
    b = [bus for (i,bus) in data["bus"] if bus["bus_type"] != 4]
    bus_ordered = sort(b, by=(x) -> x["index"])
    θ,vm = [bus["va"] for bus in bus_ordered],[bus["vm"] for bus in bus_ordered]
    return [θ ; vm]
end

"""
2nd definition (Multiple dispatch!): 
Given vector of rectangular complex voltages [e+jf], calculate the phasor form of the voltages [vangle; vmag]
"""
function calc_phasor_bus_voltage(v_rect::AbstractArray)
    θ,vm = [angle(v_i) for v_i in v_rect],[abs(v_i) for v_i in v_rect]
    return [θ ; vm]
end
