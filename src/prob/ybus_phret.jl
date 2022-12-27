struct YbusPhaseRetrievalModel
    Y::AdmittanceMatrix
    vm::AbstractArray{Real}
    va::AbstractArray{Real}
    s::AbstractArray{Complex}
end

function YbusPhaseRetrievalModel(net::Dict{String,Any})
    compute_ac_pf!(net)
    Y = calc_basic_admittance_matrix(net)
    s = calc_basic_bus_injection(net)
    v = calc_basic_bus_injection(net)
    vm,va = abs.(v),angle.(v)
    return YbusPhaseRetrievalModel(Y,vm,va,s)
end

"""
Given a YbusPhaseRetrievalModel, solve the problem.
"""
function solve_ybus_phret()

