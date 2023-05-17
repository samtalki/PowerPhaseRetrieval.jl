using DataFrames,CSV
using Distributions,Random

"""
apply updates and solve power flows to generate voltage magnitudes, active powers, and reactive powers, and jacobian structs
"""
function generate_phase_retrieval_data(net::Dict,load_updates::Vector;sel_bus_types=[1])
    @assert haskey(net,"basic_network") #must be a basic network
    # select the bus indeces
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    # setup the data containers
    vms,vas,ps,qs,jacs = [],[],[],[],[]
    for lu_t in load_updates

        # update the network
        update_data!(net,lu_t)
        compute_ac_pf!(net)
        
        # calculate the data
        vm_t = abs.(calc_basic_bus_voltage(net))[sel_bus_idx]
        va_t = angle.(calc_basic_bus_voltage(net))[sel_bus_idx]
        inj = calc_basic_bus_injection(net)[sel_bus_idx]
        p_t,q_t = real.(inj),imag.(inj)
        jac_t = PPR.calc_jacobian_matrix(net,sel_bus_types)

        # append the data
        push!(vms,vm_t)
        push!(vas,va_t)
        push!(ps,p_t)
        push!(qs,q_t)
        push!(jacs,jac_t)

    end

    return vms,vas,ps,qs,jacs
end

"""
Generates the real time va data using the 5 minute data
"""
function generate_realtime_va_data(net::Dict,load_updates::Vector;sel_bus_types=[1])
    @assert haskey(net,"basic_network") #must be a basic network
    # select the bus indeces
    sel_bus_idx = PPR.calc_bus_idx_of_type(net,sel_bus_types)
    # setup the data containers
    vas = []
    for lu_t in load_updates
        # update the network
        update_data!(net,lu_t)
        compute_ac_pf!(net)
        va_t = angle.(calc_basic_bus_voltage(net))[sel_bus_idx]
        push!(vas,va_t)
    end
    return vas
end

"""
Generate a vector of time series load datas for the network.
"""
function generate_updates(net::Dict,mw_data::DataFrame,mvar_data::DataFrame)
    m_samples,n_buses = size(mw_data)
    baseMVA = net["baseMVA"]
    @assert n_buses == length(net["bus"])
    @assert haskey(net,"basic_network") #must be a basic network
    # setup the load updates
    load_updates = [] # list of network load updates
    load_dict = net["load"] #current load dictionary
    for t=1:m_samples #for every time series sample
        load_dict_t = Dict{String,Any}()
        for (load_id,load_entry) in load_dict
            #bus_id = string(load_entry["load_bus"])
            _,bus_id = load_entry["source_id"]
            bus_id = string(bus_id)
            pd_t, qd_t = mw_data[t,bus_id]/baseMVA, mvar_data[t,bus_id]/baseMVA
            load_dict_t[load_id] = Dict("pd"=>pd_t,"qd"=>qd_t)
        end
        push!(load_updates,Dict("load"=>load_dict_t))
    end
    return load_updates
end


subsample_data(data::DataFrame,sub_sample_rate::Int=3,start_idx=1) = data[start_idx:sub_sample_rate:end,:]


"""
given high-res data, generate subsampled data with optional noise and random delays
"""
function generate_subsampled_data(data::DataFrame,
    dist_noise::Distribution=Normal(0,0.1),
    prob_delay::Float64=0.1; # probability of a delay occuring at any arbitrary node
    sub_sample_rate::Int=3,
    start_idx=1
)   
    # generate the subsampled data    
    data_sub = data[start_idx:sub_sample_rate:end,:] #subsample the data
    
    #apply noise to the data, if all of the entries in the column are not zero (otherwise, apply zero noise)
    for col in names(data_sub)
        if sum(data_sub[col]) != 0
            data_sub[col] = data_sub[col] .+ rand(dist_noise,size(data_sub[col]))
        end
    end
    
    # generate the delays
    (m_samples,n_buses) = size(data_sub)
    dist_delay = Bernoulli(prob_delay)
    delays = rand(dist_delay,n_buses)
    for (b,delay) in enumerate(delays)
        data_sub[:,b] = circshift(data_sub[:,b],delay)
    end
    
    return data_sub
end


"""
given multiple high-res datasets, generate subsampled data with optional noise and random delays
NOTE: uses the SAME DELAYS for all datasets
"""
function generate_subsampled_data(data::AbstractArray{DataFrame},
    dist_noise::Distribution=Normal(0,0.1),
    prob_delay::Float64=0.1; # probability of a delay occuring at any arbitrary node
    sub_sample_rate::Int=3,
    start_idx=1
)   
    (m_samples,n_buses) = size(data[1])
    # generate the subsampled data    
    #noise_matrix = rand(dist_noise,length(start_idx:sub_sample_rate:m_samples),n_buses)
    for (i,data_i) in enumerate(data)
        data[i] = data_i[start_idx:sub_sample_rate:end,:] #subsample the data
        noise_matrix = rand(dist_noise,size(data[i])) #generate the noise matrix
        #apply noise to the data, if all of the entries in the column are not zero (otherwise, apply zero noise)
        for (col_idx,col) in enumerate(names(data[i]))
            if !(sum(data[i][!,col]) ≈ 0)
                data[i][!,col] = data[i][!,col] .+ noise_matrix[:,col_idx]
            end
        end
    end
        
    # generate the delays
    (m_subsamples,n_buses) = size(data[1])
    dist_delay = Bernoulli(prob_delay)
    delays = rand(dist_delay,n_buses)
    for (i,data_i) in enumerate(data)
        for (b,delay) in enumerate(delays)
            data[i][:,b] = circshift(data_i[:,b],delay)
        end
    end
    
    return data
end



