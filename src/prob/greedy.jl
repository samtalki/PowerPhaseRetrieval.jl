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
    #@assert all(abs.(y0) .≈ b)
    (n,p) = size(A)
    y = y0 # initial guess
    ys = [y0] # previous guess
    for k =1:maxiter # iterate until convergence
        projection = A*pinv(A)*y
        for i = 1:n
            y[i] = b[i]*projection[i]/abs(projection[i])
        end
        push!(ys,y)
        print(string(k)*" iter: ",norm(ys[k+1]-ys[k]))
        if k > maxiter
            println("Maximum iterations reached.")
            break
        elseif k > 2 && norm(ys[k+1]-ys[k]) < ϵ
            println("Converged in $(k) iterations.")
            break
        end
    end
    return y
end

"""
Waldspurger's greedy algorithm for phase retrieval.
"""
function waldspurger(u0::Vector,b::Vector,A::Matrix,ϵ::Float64=1e-15,maxiter::Int=1000)
    (n,p) = size(A)
    @assert all([norm(uᵢ) ≈ 1 for uᵢ in u0])
    M = Diagonal(b)*(I - A*pinv(A))*Diagonal(b) # the matrix M
    u=u0 # initial guess
    u_prev = u0 # previous guess
    for k=1:maxiter
        for i = 1:n
            ΣⱼMⱼᵢūⱼ = 0
            for j=1:n
                if j != i 
                    ΣⱼMⱼᵢūⱼ += M[j,i]*conj(u[j])
                else
                    continue
                end
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
