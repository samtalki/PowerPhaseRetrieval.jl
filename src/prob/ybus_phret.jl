struct YbusPhaseRetrievalModel
    Y::AdmittanceMatrix
    vm::AbstractArray{Real}
    s::AbstractArray{Complex}
    model::Model
end

function YbusPhaseRetrievalModel(net::Dict{String,Any})
    compute_ac_pf!(net)
    Y = calc_basic_admittance_matrix(net)
    s = calc_basic_bus_injection(net)
end

