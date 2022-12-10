import Pkg
Pkg.add(LinearAlgebra,Plots,LaTeXStrings,Distributions,Random)
using LinearAlgebra,Plots,LaTeXStrings,Distributions,Random

#Two bus test case for the voltage-phase free NR power flow

#Impedance
Z=0.1

#Admittance
Y = [-10 10;
    10 -10]

#Flat start
θ_2,V_2 = 0,1

#Initial guess for state at PQ bus
x0 = [θ_2,V_2]

#load at pq bus
pd,qd = 2,1



#power balance equations
p(x) = x[2]*10*sin(x[1])
q(x) = -x[2]*10*cos(x[1]) + x[2]^2*10

"""
AC Power Flow mismatches
"""
f(x) = [
	p(x) + pd,
	q(x) + qd
]

"""
The standard Newton Raphson jacobian
"""
J(x) = [
		dpdθ(x) dpdv(x);
		dqdθ(x) dqdv(x)
]

"""
The θ-free Newton-Raphson Power Flow Jacobian
"""
Jph(x) = [
	x[2]*dqdv(x)-2*q(x) dpdv(x);
	-x[2]*dpdv(x)+2*p(x) dqdv(x)
]

"""
Given guess for initial state and tolerance ϵ solve the Newton Raphson Power Flow
"""
function nr_pf(x_0,ϵ=0.0001)
	#Arrays to store the mismatches, states, and Jacobians at each iteration.
	f_iter,x_iter,J_iter = [],[],[]
	x_k = x_0
	J_k,f_k = J(x_k),f(x_k)
	push!(x_iter,x_k)
	push!(f_iter,f_k)
	push!(J_iter,J_k)
	while norm(f(x_k)) > ϵ
		x_k -= inv(J_k)*f_k
		J_k,f_k = J(x_k),f(x_k)
		push!(x_iter,x_k)
		push!(f_iter,f_k)
		push!(J_iter,J_k)
	end
	return f_iter,x_iter,J_iter
end


"""
Given guess for initial state and tolerance ϵ solve the θ-free Newton Raphson Power Flow
"""
function θfree_nr_pf(x_0,ϵ=0.0001)
	f_iter,x_iter,J_iter = [],[],[]
	x_k = x_0
	J_k,f_k = Jph(x_k),f(x_k)
	push!(x_iter,x_k)
	push!(f_iter,f_k)
	push!(J_iter,J_k)
	while norm(f(x_k)) > ϵ
		x_k -= inv(J_k)*f_k
		J_k,f_k = Jph(x_k),f(x_k)
		push!(x_iter,x_k)
		push!(f_iter,f_k)
		push!(J_iter,J_k)
	end
	return f_iter,x_iter,J_iter
end

f_iter,x_iter,J_iter = nr_pf(x0)

fph_iter,xph_iter,Jph_iter = θfree_nr_pf(x0)


Δp,Δq = [f_i[1] for f_i in f_iter],[f_i[2] for f_i in f_iter]
Δpf,Δqf = [ffi[1] for ffi in fph_iter],[ffi[2] for ffi in fph_iter]

p1=plot(Δp,label=L"$\Delta p_k$")
plot!(Δq,label=L"$\Delta q_k$")
ylabel!("Mismatch")
title!(L"Standard $J(x_k)$")
p2= plot(Δpf,label=L"$\Delta p_k$")
plot!(Δqf,label=L"$\Delta q_k$")
title!(L"$\theta$-free $J(x_k)$")


p3 = plot([x_i[1] for x_i in x_iter],label=L"v_k")
plot!([x_i[2] for x_i in x_iter],label=L"\theta_k")
ylabel!("State")
p4 = plot([x_i[1] for x_i in xph_iter],label=L"v_k")
plot!([x_i[2] for x_i in xph_iter],label=L"\theta_k")

plot(p1,p2,p3,p4)

xlabel!(L"Iteration $k=1,..$")
savefig("2bus_iterations.pdf")
