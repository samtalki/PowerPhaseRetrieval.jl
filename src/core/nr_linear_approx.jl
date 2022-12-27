"""
Newton Raphson Power Flow data 
Supports a single iteration and all iterations.
"""
struct NRPFData
    n_bus::Int
    iter::Int #iteration number k
    rect_x::AbstractArray{Complex} #Rectangular grid state x∈C^n
    rect_f::AbstractArray{Complex} #Rectangular bus mismatches f(x) 
    delta_vm::AbstractArray{Real} #Change in voltage mag
    delta_va::AbstractArray{Real} #Change in voltage angle
    delta_q::AbstractArray{Real} #Change in reactive power
    delta_p::AbstractArray{Real} #Change in active power
end

struct NRPFLinearApprox 
    f::Function #linear approximation of the mismatches
    x::AbstractArray{Complex} #Current operating point
    f_x::AbstractArray{Complex} #Mistmaches evaluated at current operating point
    J::JacobianMatrix #Jacobian matrix
end


function calc_basic_linear_approx!(net)
    compute_ac_pf!(net)
    return calc_basic_linear_approx(net)
end
"""
Calculates a basic linear approximation of the mismatches at the current grid state.
"""
function calc_basic_linear_approx(net::Dict{String,Any})
    J = calc_jacobian_matrix(net)

end





"""
Given complex voltage and powers, and an admittance matrix Y, calculate the power flow mismatch.
"""
function calc_basic_mismatch(V,S,Y)
    # STEP 1: Compute mismatch and check convergence
    Si = V .* conj(Y * V)
    Δp, Δq = real(S - Si), imag(S - Si)
    Δx = [Δp ; Δq]
    return Δx
end

