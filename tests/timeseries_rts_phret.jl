include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
using PowerModels,LinearAlgebra,Plots,DataFrames,CSV

# load case and run power flow
net = make_basic_network(parse_file("data/networks/case_RTS_GMLC.m"))
compute_ac_pf!(net)

# load the averaged timeseries data
mvar_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mvar_data.csv"))
mw_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mw_data.csv"))

# generate random delay
function generate_delay()


end