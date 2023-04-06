using LinearAlgebra


"""
The Gerchberg Saxton algorithm for phase retrieval.

Given:
    y0 - initial guess for the phasor
    b - vector of magnitude measurements
    A - tall or wide design matrix
    ϵ - tolerance for convergence

Returns:
    y - the converged phasor
"""
function gerchberg_saxton(y0::Vector,b::Vector,A::Matrix,ϵ::Float64=1e-15,maxiter::Int=1000)
    @assert all(abs.(y0) .≈ b)
    (n,p) = size(A)
    y = y0 # initial guess
    y_prev = y0 # previous guess
    for k =1:maxiter # iterate until convergence
        projection = A*pinv(A)*y_prev
        for i = 1:n
            y[i] = b[i]*projection[i]/abs(projection[i])
        end
        if k > maxiter
            println("Maximum iterations reached.")
            break
        # elseif norm(y-y_prev) < ϵ
        #     println("Converged in $(k) iterations.")
        #     break
        end
        y_prev = y
    end
    return y
end

"""
Waldspurger's greedy algorithm for phase retrieval.
"""
function waldspurger(u0::Vector,b::Vector,A::Matrix,ϵ::Float64=1e-15,maxiter::Int=1000)
    (n,p) = size(A)
    @assert all([norm(uᵢ) == 1 for uᵢ in u0])
    M = Diagonal(b)*(Id - A*pinv(A))*Diagonal(b) # the matrix M
    u=u0 # initial guess
    u_prev = u0 # previous guess
    for k=1:maxiter
        for i = 1:n
            ΣⱼMⱼᵢūⱼ = 0
            for j=1:n
                j != i || continue
                ΣⱼMⱼᵢūⱼ += M[j,i]*conj(u[j])
            end
            u[i] = -ΣⱼMⱼᵢūⱼ/abs(ΣⱼMⱼᵢūⱼ)
        end
        if k > maxiter
            println("Maximum iterations reached.")
            break
        # elseif norm(u-u_prev) < ϵ
        #     println("Converged in $(k) iterations.")
        #     break
        end 
        u_prev = u
    end 
    return u
end
