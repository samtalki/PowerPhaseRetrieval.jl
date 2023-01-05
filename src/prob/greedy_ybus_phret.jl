# =====
# Greedy algorithm
# =====

struct GreedyYbusPhretSolution
    data::YbusPhretData
    u_hat::Vector
    th_hat::Vector
    irect_hat::Vector
    rel_err::Float64
    u_iter::Vector{Vector{ComplexF64}} #iterations of the greedy algorithm
end
"""
Solve the greedy ybus phasecut problem from "Phase Recovery, Maxcut and Complex Semidefinite Programming"
"""
function solve_greedy_ybus_phasecut(data::YbusPhretData;sigma_init_phase=0.1,max_iter=5,ϵ=1e-9)
    n_bus,Y,Vmag,Imag = data.n_bus,data.Y,data.Vmag,data.Imag #unpack data
    M = Diagonal(Imag)*(I - Y*pinv(Y))*Diagonal(Imag)
    #initialize the phase
    dist_uph = π/2*Normal(0,sigma_init_phase)
    init_Th = rand(dist_uph,n_bus)
    u = [Imag_i*cos(θ_i) + Imag_i*sin(θ_i)*im for (Imag_i,θ_i) in zip(Imag,init_Th)]
    u = [u_i/abs(u_i) for u_i in u]
    #u_iter = zeros(max_iter,length(u))
    #u_iter[1,:] = u
    #while (k>1 && norm(u_iter[k]-norm(u_iter[k-1]))>ϵ) && k<=max_iter
    #k=1
    for k=1:max_iter
        for i=1:n_bus 
            off_diag_sum = sum([M[j,i]*u[j] for j=1:n_bus if j !== i])
            if Imag[i] < 1e-3 || norm(u[i])<1e-6
                u[i] = 1e-6 + 1e-6*im
                continue
            end
            u[i] = (-off_diag_sum)/(abs(off_diag_sum))
        end
        #k += 1
    end
    return u
    #u_hat = u_iter[end]
   # @assert all([abs(u_i)==1 for u_i in u_iter[end]])
    # return GreedyYbusPhretSolution(
    #     data,
    #     u_iter[end],
    #     angle.(u_iter[end]),
    #     Diagonal(Imag)*u_hat,
    #     norm(angle.(u_hat)-data.Iangle)/norm(data.Iangle)*100,
    #     u_iter
    # )
end

function solve_greedy_ybus_phasecut!(net::Dict{String,Any};sigma_init_phase=0.1,max_iter=1000,ϵ=1e-9)
    data = YbusPhretData(net)
    u_iter = solve_greedy_ybus_phasecut(data,sigma_init_phase=sigma_init_phase,max_iter=max_iter,ϵ=ϵ)
    return u_iter
end
