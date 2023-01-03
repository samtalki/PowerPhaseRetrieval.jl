include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM
using JuMP,SCS,LinearAlgebra

net = PM.make_basic_network(PM.parse_file("/home/sam/github/PowerSensitivities.jl/data/radial_test/case14.m"))
PM.compute_ac_pf!(net)
idx = PPR.calc_bus_idx_of_type(net,[1])
Y = Matrix(PM.calc_basic_admittance_matrix(net))
Yr,Yi = real.(Y),imag.(Y)
T_Y = [Yr -1*Yi;
        Yi Yr]

Yprime = Y[idx,idx]
Vrect = PM.calc_basic_bus_voltage(net)
Vrectprime = Vrect[idx]
S = PM.calc_basic_bus_injection(net)
Sprime = S[idx]

Im = abs.(S) ./ abs.(Vrect)
Imprime = abs.(Sprime) ./ abs.(Vrectprime)

Irect = Y*Vrect
Irectprime = Yprime*Vrectprime
Imprime_true = abs.(Irectprime)


n_bus,Y,Vmag,Imag = data.n_bus,data.Y,data.Vmag,data.Imag #unpack data
Iangle,Vangle = data.Iangle, data.Vangle


#Bus type idx
pq_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[1])
pv_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[2])
pq_pv_idx = sort(pq_idx ∪ pv_idx)
slack_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[3])


model = Model(SCS.Optimizer)
#set_optimizer_attribute(model, "max_iter", 6000)
#Complex voltage
@variable(model,vr[1:n_bus])
@variable(model,vi[1:n_bus])

#Complex current phase
@variable(model,ur[1:n_bus])
@variable(model,ui[1:n_bus])

#PSD matrix
@variable(model,X[1:2*n_bus,1:2*n_bus], PSD)

#Construct M matrix (fixed phase representation)
M = Diagonal(Imag)*(I - Y * pinv(Y))*Diagonal(Imag)
#transformed M matrix
Mr,Mi = real.(M),imag.(M)
@expression(model,T_M,
    [
        Mr -1*Mi; 
        Mi Mr
    ]
)

#PQ/PV BUS Constraints
for k=1:n_bus
    if k ∈ pv_idx || k ∈ slack_idx
        for j =1:n_bus
            u_k = cos(Iangle[k]) + sin(Iangle[k])*im
            u_j = cos(Iangle[j]) + sin(Iangle[j])*im

            Ukj = u_k*conj(transpose(u_j))
            Real_Ukj = real(Ukj)
            Imag_Ukj = imag(Ukj)

            @constraint(model,X[k,j]== Real_Ukj)
            @constraint(model,X[k+n_bus,j] == Imag_Ukj)
            @constraint(model,X[k,j+n_bus] == -Imag_Ukj)
            @constraint(model,X[k+n_bus,j+n_bus] == Real_Ukj)
        end
        
    end
end

#Complex constraints
for i = 1:n_bus
    @constraint(model,X[i,i] ==1)
    @constraint(model,X[i+n_bus,i+n_bus]==1)
    #@constraint(model,X[1:n_bus,i] .== ur*ur[i])
    #@constraint(model,X[n_bus+1:end,i] .== ui*ur[i])
    #@constraint(model,X[1:n_bus,i+n_bus] .== -ur.*ui[i])
    #@constraint(model,X[n_bus+1:end,i+n_bus] .== ui*ui[i])
end

#Trace objective
@objective(
    model,
    Min,
    tr(T_M*X)
)
optimize!(model)

#Extract solution, check for uncertainty of the relaxation
X_opt = value.(X)[1:n_bus,1:n_bus] + value.(X[1+n_bus:end,1:n_bus]) .*im
#X_opt = value.(X)
# evals = eigvals(X_opt)
# if length([e for e in evals if abs(real(e)) > 1e-4]) >1
#     X_hat = calc_closest_rank_r(X_opt,1)
# end
#X_hat = X_opt[1:n_bus,1:n_bus] .+ X_opt[n_bus+1:end,1:n_bus] .* im
# evals,evecs = eigen(X_opt)
# uhat = evecs[:,1] ./ abs.(evecs[:,1])

# #reconstruct the current phase, voltage phase, etc.
# Iangle_est = angle.(uhat)
# Irect_est = Diagonal(Imag)*uhat
# #Irect_est = Imag .* (cos.(Iangle_est) + sin.(Iangle_est)*im)
# Vrect_est = pinv(Y)*Diagonal(Imag)*uhat
# Vangle_est = angle.(Vrect_est)


#greedy_results = PPR.solve_greedy_ybus_phasecut!(net)
