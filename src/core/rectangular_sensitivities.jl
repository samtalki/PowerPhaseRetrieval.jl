function calc_rectangular_jacobian_sensitivities(G,B,v)
    _,n = size(G)
    ∂pe = zeros(n,n)
    ∂qe = zeros(n,n)
    ∂pf = zeros(n,n)
    ∂qf = zeros(n,n)
    ∂v²e = zeros(n,n)
    ∂v²f = zeros(n,n)
    for i = 1:n
        for k=1:n
            if k != i # off-diagonal elements
                ∂pe[i,k] = G[i,k]*real(v[i]) + B[i,k]*imag(v[i])
                ∂pf[i,k] = G[i,k]*imag(v[i]) - B[i,k]*real(v[i])
                ∂qe[i,k] = ∂pf[i,k]
                ∂qf[i,k] = -∂pe[i,k]
                ∂v²e[i,k] = 0
                ∂v²f[i,k] = 0

            else # diagonal elements
                eᵢ,fᵢ = real(v[i]),imag(v[i])
                #--- active power to  complex voltage
                ∂pe[i,i] = sum(
                    [G[i,k]*real(v[k])-B[i,k]*imag(v[k]) for k=1:n]
                ) + G[i,i]*eᵢ + B[i,i]*fᵢ
                ∂pf[i,i] = sum(
                    [G[i,k]*imag(v[k])+B[i,k]*real(v[k]) for k=1:n]
                ) - B[i,i]*eᵢ + G[i,i]*fᵢ 

                #--- reactive power to complex voltage
                ∂qe[i,i] = sum(
                    [-G[i,k]*imag(v[k]) - B[i,k]*real(v[k]) for k=1:n]
                ) - B[i,i]*eᵢ + G[i,i]*fᵢ
                ∂qf[i,i] = sum(
                    [G[i,k]*real(v[k]) - B[i,k]*imag(v[k]) for k=1:n]
                ) - G[i,i]*eᵢ - B[i,i]*fᵢ

                #--- squared voltage to complex voltage
                ∂v²e[i,i] = 2*eᵢ
                ∂v²f[i,i] = 2*fᵢ
            end
        end
    end 
    return ∂pe,∂qe,∂pf,∂qf,∂v²e,∂v²f
end