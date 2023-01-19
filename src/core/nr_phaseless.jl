"""
Given a PowerModels network and a phase retrieval model, 
Solves the power flow equations with Newton-Raphson power flow,
Solving the phase retrieval problem at every iteration.
"""
function calc_phaseless_nr_pf!(
    net::Dict{String,Any},phret_model::AbstractPhaseRetrievalModel;
    tol=1e-4,itr_max=20)
    data = calc_nr_pf!(net)
    for (k,(f_k,x_k)) in enumerate(zip(data.f,data.x))
        
    end

end