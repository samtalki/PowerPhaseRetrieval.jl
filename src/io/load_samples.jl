using DataFrames,PowerModels


"""
Time series- AMI-data
"""
struct AMIData
    net::Dict{String,Any} #PowerModels network
    Vm::AbstractMatrix #matrix of voltage magnitudes
    Va::AbstractMatrix #matrix of voltage phase angles
    P::AbstractMatrix #matrix of net active power injections
    Q::AbstractMatrix #matrix of net reactive power injections
    J_nom::PowerFlowJacobian #nominal power flow jacobian at default power flow solution (estimated should be close)
end

"""
Given a power models data dict and an OPF learn dataset construct AMIData
"""
function AMIData(net::Dict{String,Any},df::DataFrame)
    compute_ac_pf!(net)
    J_nom = calc_jacobian_matrix(net,[1]) #nominal jacobian
    
    #get number of buses
    n_bus = length(net["bus"])

    #get number of PQ buses
    pq_bus_idx = calc_bus_idx_of_type(net,[1])
    n_pq_bus = length(pq_bus_idx)

    #get load idx mapping to pq bus idx
    load_idx_to_pq_bus_idx = get_load_pq_bus_mapping(net)
    n_loads = length(load_idx_to_pq_bus_idx)
    

    #initialize time series matrices
    M_samples = size(df)[1]
    Vm,Va,P,Q = zeros(M_samples,n_bus),zeros(M_samples,n_bus),zeros(M_samples,n_bus),zeros(M_samples,n_bus)

    for bus_idx=1:n_pq_bus
        #Get the voltage magnitude and phase angle at every pq bus 
        v_bus = conj.(parse.(Complex{Float64},df[!,"bus"*string(bus_idx)*":v_bus"])) #NOTE: THE NREL DATASETS NEED TO BE CONJUGATED
        Vm[:,bus_idx],Va[:,bus_idx] = abs.(v_bus),angle.(v_bus)

        #Get the net injections for every PQ bus
        for (load_idx,load_bus_idx) in load_idx_to_pq_bus_idx
            #Save the net injections
            P[:,load_bus_idx] .+= -1 .* df[!,"load"*string(load_idx)*":pl"] 
            Q[:,load_bus_idx] .+= -1 .* df[!,"load"*string(load_idx)*":ql"]
        end
    end
    
    return AMIData(
        net,
        Vm[:,pq_bus_idx],
        Va[:,pq_bus_idx],
        P[:,pq_bus_idx],
        Q[:,pq_bus_idx],
        J_nom
    )
end

"""
Get two dictionaries that map loads and generators to PQ bus indeces (every bus with a load)
"""
function get_load_pq_bus_mapping(net::Dict{String,Any})
    loads = net["load"]
    n_load = length(loads)
    load_bus_idx = Dict()
    for load_idx=1:n_load
        load_bus_idx[string(load_idx)] = loads[string(load_idx)]["load_bus"]
    end
    return load_bus_idx
end



"""
Struct for tensors of estimated jacobians
"""
struct EstimatedJacobians
    DPDV::Array{Float64,2} # NxNxM tensor of estimated active-voltage sensitivities
    DQDV::Array{Float64,2}
    DPDTH::Array{Float64,2}
    DQDTH::Array{Float64,2}
    rel_err::Dict{String,Float64} #relative errors of each jacobian
end

"""
Compute the timeseries of esitmated jacobians.
"""
function EstimatedJacobians(ami::AMIData)
    #extract AMI data
    Vm,Va,P,Q = ami.Vm,ami.Va,ami.P,ami.Q
    
    #Make finite differences
    dVm,dVa,dP,dQ = [diff(M,dims=1) for M in (Vm,Va,P,Q)]
    m_diff,n_bus = size(dVm)

    #Initialize estimated jacobian tensors
    DPDV,DQDV,DPDTH,DQDTH = [zeros(n_bus,n_bus) for i=1:4]

    #find S: dP S = dV
    DPDV = pinv(P)*Vm
    DQDV = pinv(Q)*Vm
    DPDTH = pinv(P)*Va
    DQDTH = pinv(Q)*Va

    #compute rel err
    err(xhat,xtrue) = norm(xhat-xtrue)/norm(xtrue)
    rel_err = Dict(
        "pv" => err(DPDV,ami.J_nom.pv),
        "qv" => err(DQDV,ami.J_nom.qv),
        "pth" => err(DPDTH,ami.J_nom.pth),
        "qth" => err(DQDTH,ami.J_nom.qth)
    )

    return EstimatedJacobians(
        DPDV,
        DQDV,
        DPDTH,
        DQDTH,
        rel_err
    )
end

"""
Given power models dict and timeseries make the estimated jacobians
"""
function EstimatedJacobians(net::Dict{String,Any},df::DataFrame)
    ami = AMIData(net,df)
    return EstimatedJacobians(ami)
end


