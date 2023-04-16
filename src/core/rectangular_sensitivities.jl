function calc_rectangular_jacobian_sensitivities(G,B,v)
    _,n = size(Y)
    ∂pe = zeros(ComplexF64,n,n)
    ∂qe = zeros(ComplexF64,n,n)
    ∂pf = zeros(ComplexF64,n,n)
    ∂qf = zeros(ComplexF64,n,n)
    ∂v²e = zeros(ComplexF64,n,n)
    ∂v²f = zeros(ComplexF64,n,n)
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

                #--- active power to  complex voltage
                ∂pe[i,i] = sum(
                    [G[i,k]*real(v[k])-B[i,k]*imag(v[k]) for k=1:n]
                ) + G[i,i]*real(v[i]) + B[i,i]*imag(v[i])
                ∂pf[i,i] = sum(
                    [G[i,k]*imag(v[k])+B[i,k]*real(v[k]) for k=1:n]
                ) + G[i,i]*imag(v[i]) - B[i,i]*real(v[i])

                #--- reactive power to complex voltage
                ∂qe[i,i] = sum(
                    [-G[i,k]*imag(v[k]) - B[i,k]*real(v[k]) for k=1:n]
                ) - B[i,i]*real(v[i]) + G[i,i]*imag(v[i])
                ∂qf[i,i] = sum(
                    [G[i,k]*real(v[k]) - B[i,k]*imag(v[k]) for k=1:n]
                ) - G[i,i]*real(v[i]) - B[i,i]*imag(v[i])

                #--- squared voltage to complex voltage
                ∂v²e[i,i] = 2*real(v[i])
                ∂v²f[i,i] = 2*imag(v[i])
            end
        end
    end 
    return ∂pe,∂qe,∂pf,∂qf,∂v²e,∂v²f
end