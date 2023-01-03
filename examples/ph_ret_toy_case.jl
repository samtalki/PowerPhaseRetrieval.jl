include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
using PowerModels
using LinearAlgebra,Plots
import SCS
using Convex
#rts test case
net = make_basic_network(parse_file("/home/sam/github/PowerSensitivities.jl/data/pm_matpower/case30.m"))
results = PPR.sdp_sens_phret(net,sel_bus_types=[1])
th_hat,th_true = results["va_hat"],results["va_true"]
err = norm(th_hat - th_true)/norm(th_true)*100
#maxcut_results = PPR.maxcut_sens_phret(net)


n = 20
p = 2
A = rand(n,p) + im*randn(n,p)
x = rand(p) + im*randn(p)
b = abs.(A*x) + rand(n)
b_rect = A*x
u_true = b_rect ./ abs.(b_rect)

M = diagm(b)*(I(n)-A*pinv(A))*diagm(b)
U = Convex.ComplexVariable(n,n)
objective = inner_product(U,M)
c1 = diag(U) == 1
c2 = U in :SDP
p = minimize(objective,c1,c2)
solve!(p, () -> SCS.Optimizer())
U_hat = evaluate(U)
U_true = u_true*conj(u_true')

B, C = eigen(U_hat)

u = C[:,1];
for i in 1:n
    u[i] = u[i]/abs(u[i])
end

b_angle_hat = angle.(u)
b_angle_true = angle.(A*x)
relerr = norm(b_angle_true-b_angle_hat)/norm(b_angle_true)
x_angle = atan.(imag.(x) ./ real.(x))