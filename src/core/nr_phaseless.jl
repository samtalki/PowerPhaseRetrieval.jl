"""
Given a PowerModels network and a phase retrieval model, 
Solves the power flow equations with Newton-Raphson power flow,
Solving the phase retrieval problem at every iteration.
"""
function calc_phaseless_nr_pf!(net::Dict{String,Any},phret_model::AbstractPhaseRetrievalModel;tol=1e-4,itr_max=20)
    data = calc_nr_pf!(net)
    return calc_phaseless_nr_pf(data,phret_model,tol=tol,itr_max=itr_max)

end

function calc_phaseless_nr_pf(data::NRPFData,phret_model::AbstractPhaseRetrievalModel;tol=1e-4,itr_max=20)
    f,x,S,L = data.f,data.x,data.S_inj,data.L_inj
    jacs = data.jacobians
    for (k,(f_k,x_k,S_k,L_k)) in enumerate(zip(f,x,S,L))
        vk,θk = abs.(x_k),angle.(x_k)
        
    end
end