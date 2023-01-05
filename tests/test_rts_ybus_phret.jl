# ------------- 
# Working Ybus phret algorithm with SDP
# -------------
net = PM.make_basic_network(PM.parse_file("/home/sam/github/PowerSensitivities.jl/data/radial_test/case24_ieee_rts.m"))
PM.compute_ac_pf!(net)
n_bus = length(net["bus"])

Y = Matrix(PM.calc_basic_admittance_matrix(net))
Vrect = PM.calc_basic_bus_voltage(net)
Vangle = angle.(Vrect)
S = PM.calc_basic_bus_injection(net)
Imag = abs.(Y*Vrect) #abs.(S) ./ abs.(Vrect)

Irect = Y*Vrect
Iangle = angle.(Irect)
Imag_true = abs.(Irect)
@assert Diagonal(Imag_true)*cis.(Iangle) ≈ Irect ≈ Diagonal(Imag)*cis.(Iangle)

#Bus type idx
slack_idx = PPR.PowerSensitivities.calc_bus_idx_of_type(net,[3])

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
    if k ∈ slack_idx
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
Irect_est = Diagonal(Imag)*cis.(Iangle_est)
Irest_est_shift = Diagonal(Imag)*cispi.(Iangle_est)
Vrect_est = inv(Y)*Irect_est
Vangle_est = angle.(Vrect_est)
Vrect_est_shift = inv(Y)*Irest_est_shift

rel_err(Ahat,Atrue) = string(norm(Ahat-Atrue)/norm(Atrue)*100)
println("Vrect rel_err: "*rel_err(Vrect_est,Vrect))
println("Vrect_shift rel_err: "*rel_err(Vrect_est_shift,Vrect))
println("Irect_est rel_err: "*rel_err(Irect_est,Irect))
println("Irect_est_shift rel_err: "*rel_err(Irest_est_shift,Irect))
println("Vangle_est rel err: "*rel_err(Vangle_est,Vangle))