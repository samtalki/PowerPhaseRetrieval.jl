using PowerModels
using JuMP 
using Ipopt
using LinearAlgebra


net = make_basic_network(parse_file("../data/case_RTS_GMLC.m"))
compute_ac_pf!(net)
