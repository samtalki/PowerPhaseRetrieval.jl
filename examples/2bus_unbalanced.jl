using LinearAlgebra,ForwardDiff
using Plots,LaTeXStrings

# ----- Two bus test case for the unbalanced voltage-phase free Jacobian
n =2

# Impedance from bus 1 to 2
r12,x12 = 2.15, 1.55
z12 = r12 + x12*im

# Admittance
y12 = 1/z12
Y = zeros(3*n,3*n)


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

