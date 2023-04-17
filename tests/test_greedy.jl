include("../src/PowerPhaseRetrieval.jl")
include("../src/core/rectangular_sensitivities.jl")
include("../src/prob/greedy.jl")
import .PowerPhaseRetrieval as PPR
using LinearAlgebra,PowerModels
using Statistics,Random
using Plots,LaTeXStrings

# ---------------- load case and run power flow
net = make_basic_network(parse_file("data/networks/case_RTS_GMLC.m")) #basic network
pq_buses = PPR.calc_bus_idx_of_type(net,[1])
compute_ac_pf!(net)

# ----------------- get the initial values
v0 = calc_basic_bus_voltage(net)[pq_buses]
vm0 = abs.(calc_basic_bus_voltage(net)[pq_buses])
p0 = real.(calc_basic_bus_injection(net)[pq_buses])
q0 = imag.(calc_basic_bus_injection(net)[pq_buses])
e0,f0 = real.(v0),imag.(v0) # real and imaginary part of the voltage


# ----------------- get the topology data
Y = calc_basic_admittance_matrix(net)[pq_buses,pq_buses]
G,B = real.(Y),imag.(Y)

# ----------------- get the sensitivities
∂pe,∂qe,∂pf,∂qf,∂v²e,∂v²f = calc_rectangular_jacobian_sensitivities(G,B,v0)
J = [
    ∂pe + im.*∂pf;
    ∂qe + im.*∂qf;
    ∂v²e + im.*∂v²f
]
m,_ = size(J)
n = length(v0)


# ----------------- get the measurements
y_true = zeros(ComplexF64,m)
y_true[[p0;q0;vm0.^2] .!= 0.0] .= (J*v0)[[p0;q0;vm0.^2] .!= 0.0]
x_true = pinv(J)*y_true
b_meas = abs.(y_true) + 0.01*randn(m)
p_meas,q_meas,vm²_meas = b_meas[1:n],b_meas[n+1:2n],b_meas[2n+1:end]

# ----------------- get the initial guess
u0 = randn(ComplexF64,m)
u0 = u0./ abs.(u0)
@assert all(abs.(u0) .≈ 1)
y0 = b_meas.* u0 # initial guess for the complex voltage

# ----------------- solve the greedy problem
yhat_gs = gerchberg_saxton(y0,b_meas,J)
uhat_walds = waldspurger(u0,b_meas,J)
yhat_walds = y_meas.* uhat_walds

# -------- get the voltage phasors
xhat_gs = pinv(J)*yhat_gs
xhat_walds = pinv(J)*yhat_walds


# -------- get the forward error
err_gs = norm(yhat_gs - y_true)/norm(y_true)
err_walds = norm(yhat_walds - y_true)/norm(y_true)
println("Forward error:")
println("Error of Gerchberg-Saxton: $(err_gs)")
println("Error of Waldspurger: $(err_walds)")


# -------- get the backward error
err_gs = norm(xhat_gs - x_true)/norm(x_true)
err_walds = norm(xhat_walds - x_true)/norm(x_true)
println("Backward error:")
println("Error of Gerchberg-Saxton: $(err_gs)")
println("Error of Waldspurger: $(err_walds)")