

struct NRPFLinearApprox 
    f::Function #linear approximation of the mismatches
    x::AbstractArray{Complex} #Current operating point
    f_x::AbstractArray{Complex} #Mistmaches evaluated at current operating point
    J::PowerFlowJacobian #Jacobian matrix
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

