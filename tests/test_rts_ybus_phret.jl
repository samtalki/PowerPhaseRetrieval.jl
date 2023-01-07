include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM
using LinearAlgebra,Plots
using Random,Statistics,Distributions
using JuMP,SCS

sigma_noise = 0.001
# ------------- 
# Working Ybus phret algorithm with SDP
# -------------
net = PM.make_basic_network(PM.parse_file("/home/sam/github/PowerSensitivities.jl/data/radial_test/case_ieee30.m"))
PM.compute_ac_pf!(net)
n_bus = length(net["bus"])

Y = Matrix(PM.calc_basic_admittance_matrix(net))
Vrect = PM.calc_basic_bus_voltage(net)
Vangle = angle.(Vrect)
S = PM.calc_basic_bus_injection(net)
Imag = abs.(Y*Vrect) + rand(Normal(0,sigma_noise),n_bus)#abs.(S) ./ abs.(Vrect)

Irect = Y*Vrect
Iangle = angle.(Irect)
Imag_true = abs.(Irect)

#Samnity checks
@assert Diagonal(Imag_true)*cis.(Iangle) ≈ Irect 
@assert all(Vrect .≈ inv(Y)*Irect)
@assert all(Vrect .≈ inv(Y)*Diagonal(Imag_true)*cis.(Iangle))

#Bus type idx
slack_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[3])
pv_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[2])

model = Model(SCS.Optimizer)
set_silent(model)

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
    if k ∈ slack_idx || k ∈ pv_idx
        for j =1:n_bus
            u_k = cis(Iangle[k]) #cos(Iangle[k]) + sin(Iangle[k])*im
            u_j = cis(Iangle[j]) #cos(Iangle[j]) + sin(Iangle[j])*im

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
Xr1 = PPR.calc_closest_rank_r(X_opt,1)
u = Xr1[:,1]
Iangle_est = angle.(u)
@assert all(Diagonal(Imag)*(u./abs.(u)) .≈ Diagonal(Imag)*cis.(Iangle_est))

#Set slack/pv bus angles
#Iangle_est[slack_idx] = Iangle[slack_idx]
#Iangle_est[pv_idx] = Iangle[pv_idx]

ϕ_offset = abs.(Iangle[slack_idx] - Iangle_est[slack_idx])
Iangle_est = Iangle_est .+ ϕ_offset 

Irect_est = Diagonal(Imag)*cis.(Iangle_est)

#Correct leading/lagging power factors
# lead_lag = sign.(Irect)
# global n_pf_wrong = 0
# for (i,(hat_I,I)) in enumerate(zip(Irect_est,Irect))
#     if sign(imag(hat_I)) !== sign(imag(I))
#         Irect_est[i] = conj(Irect_est[i])
#         global n_pf_wrong+=1
#     end
# end
# println("n_pf_wrong: ",n_pf_wrong)


#--- Set slacks values
#Irect_est[slack_idx] = Irect[slack_idx]
#Irect_est[pv_idx] = Irect[pv_idx]

Vrect_est = Y \ Irect_est 

#Vrect_est[slack_idx] = Vrect[slack_idx]
#Vrect_est[pv_idx] = Vrect[pv_idx]

Vangle_est_aff = angle.(Vrect_est) .+ Iangle[1]
Vangle_est = angle.(Vrect_est)
#θ_offset = mean(Vangle[slack_idx] .- Vangle_est[slack_idx])
#Vangle_est = θ_offset .+ Vangle_est
#Vangle_est = Vangle_est .+ θ_hat_slack

rel_err(Ahat,Atrue) = string(norm(Ahat-Atrue)/norm(Atrue)*100)
println("Vrect rel_err: "*rel_err(Vrect_est,Vrect))
println("Irect_est rel_err: "*rel_err(Irect_est,Irect))
println("Vangle_est rel err: "*rel_err(Vangle_est,Vangle))
println("Iangle_est rel_err: "*rel_err(Iangle_est,Iangle))

plot([Vangle Vangle_est Vangle_est_aff],label=["true" "est." "est. aff."],marker=[:circle :square :diamond])