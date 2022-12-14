struct TaylorSeriesLinearApprox
    dp::AbstractArray
    dq::AbstractArray
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

