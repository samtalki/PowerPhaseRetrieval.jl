using PowerModels
using DataFrames
using CSV


# Load the RTS-GMLC network model
rts = make_basic_network(parse_file("../data/case_RTS_GMLC.m"))
day_ahead_data = DataFrame(CSV.File(input))