"""
Fixed point linearization of the power flow manifold described in:

S. Bolognani and F. Dörfler, "Fast power system analysis via implicit linearization of the power flow manifold," 
2015 53rd Annual Allerton Conference on Communication, Control, and Computing (Allerton), 2015, pp. 402-409, doi: 10.1109/ALLERTON.2015.7447032.
"""
using PowerModels
using LinearAlgebra

struct FixedPoint
    u::AbstractArray{Complex} #complex voltages
    vm::AbstractArray{Real} #voltage magnitudes
    va::AbstractArray{Real} #voltage angles
    s::AbstractArray{Complex} #complex power injections
    p::AbstractArray{Real} #active injections
    q::AbstractArray{Real} #reactive injections
    J::AbstractMatrix #power flow jacobian at pf solution
    A::AbstractMatrix #Linear approximation matrix provided by "Fast Power System Analysis via Implicit Linearization of the Power Flow Manifold"
end

function FixedPoint(network::Dict)
    n = length(network["bus"])
    u,s = calc_basic_bus_voltage(network),calc_basic_bus_injection(network)
    vm,va = abs.(u),angle.(u)
    p,q =  real.(s),imag.(s)
    J = calc_basic_jacobian_matrix(network)
    Y = calc_basic_admittance_matrix(network)
    N = [
        I zeros(n,n);
        zeros(n,n) -I
    ]

    R(u) = [
        Diagonal(cos.(angle.(u))) -Diagonal(abs.(u))*Diagonal(sin.(angle.(u)));
        Diagonal(sin.(angle.(u))) Diagonal(abs.(u))*Diagonal(cos.(angle.(u)))
    ]

    A = (T(Diagonal(conj.(Y*u))) + T(Diagonal(conj.(u)))*N*T)*R(conj.(u)) - I
    
    return FixedPoint(u,vm,va,s,p,q,J,A)
end

"""
Given a complex valued matrix A, transform to 
[
    ReA -ImA;
    ImA ReA
]
"""
function T(A)
    ReA,ImA = real.(A),imag.(A)
    return [
        ReA -ImA;
        ImA ReA
    ]
end


function calc_fixed_point!(network::Dict)
    compute_ac_pf!(network)
    return FixedPoint(network)
end

function calc_jac_fixed_power_comparison(network::Dict)
    fp = calc_fixed_point!(network)
    J,A = fp.J,fp.A
    return norm(A-J)/norm(J)*100
end