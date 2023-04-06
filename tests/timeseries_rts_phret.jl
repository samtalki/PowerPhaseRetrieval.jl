# create synthetic data for one day at 15 minute granularity driven by smart meter data varying demands. 
# Then assign random delays within a 15 minute range (e.g. uniform distribution). See the impact of lack of synchronization in the data stream. 
# My strong believe is that the method will be robust enough to generate the matrices and phase retrieval that have small error. 

include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
using PowerModels,LinearAlgebra,Plots
using DataFrames,CSV
using Distributions,Random

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
Generates random delays for real-time data
"""
function generate_delays()


end


"""
Given vectors of voltage magnitudes, active powers, reactive powers, and jacobian models
solve the jac recovery phase retrieval problem sequentially
"""
function test_sequential_est_bus_voltage_phase(vms,vas,ps,qs,jacs;D_f_x=nothing,D_vm=nothing)
    phret_results = []
    n_buses = length(vms[1])
    for (vm_t,va_t,p_t,q_t,jac_t) in zip(vms,vas,ps,qs,jacs)
        # generate the noise at this sample
        ϵ_f_x = D_f_x !== nothing ? rand(D_f_x,2*n_buses) : 0
        ϵ_vm = D_vm !== nothing ? rand(D_vm,n_buses) : 0
        # solve the phase retrieval problem
        est_results = PPR.est_bus_voltage_phase( #va_nom,∂pv,∂qv,vm_obs,f_x_obs,q_nom,p_nom,∂pθ_true,∂qθ_true
            va_t,
            Matrix(jac_t.pv), #∂pv
            Matrix(jac_t.qv), #∂qv
            vm_t .+ ϵ_vm, #vm_obs
            Matrix(jac_t.matrix)*[va_t;vm_t] .+ ϵ_f_x, #f_x_obs
            q_t, #q_nom
            p_t, #p_nom
            Matrix(jac_t.pth), #∂pth
            Matrix(jac_t.qth) #∂qth
            )
        push!(phret_results,est_results)
    end
    return phret_results
end

"""
Calculates the root mean squared error of the estimated voltage phase angles estimated in each hourly interval
with the true phase angles using the 5minute data.
"""
function calc_hourly_rms_error(hourly_phret_results::Vector,realtime_phase_angles::Vector)
    @assert length(realtime_phase_angles) == 12*length(hourly_phret_results)
    hourly_rms_error = []
    for i=1:length(hourly_phret_results)
        # calculate the hourly rms error
        hourly_rms_error_t = 0
        est_va_t = hourly_phret_results[i]
        for j=1:12
            t = (i-1)*12 + j
            true_va_t = realtime_phase_angles[t]
            hourly_rms_error_t += sum((est_va_t - true_va_t).^2)
        end
        hourly_rms_error_t = sqrt(hourly_rms_error_t/12)
        push!(hourly_rms_error,hourly_rms_error_t)
    end
    return hourly_rms_error
end


# load case and run power flow
net = make_basic_network(parse_file("data/networks/case_RTS_GMLC.m")) #basic network
pq_buses = PPR.calc_bus_idx_of_type(net,[1])
compute_ac_pf!(net)
p0,q0 = real.(calc_basic_bus_injection(net)[pq_buses]),imag.(calc_basic_bus_injection(net)[pq_buses])

# -----------------
# generate m_samples of timeseries data 
# ----------------
m_hours = 48 # number of hours to generate


#---------------- generate the hourly power flow result data
#--- load the averaged timeseries data
hourly_mvar_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mvar_data.csv"))
hourly_mw_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mw_data.csv"))

# remove time index column
time_idx,hourly_mw_data = hourly_mw_data[1:m_hours,1],hourly_mw_data[1:m_hours,2:end]
time_idx,hourly_mvar_data = hourly_mvar_data[1:m_hours,1],hourly_mvar_data[1:m_hours,2:end]

#--- generate the hourly load updates and grid states
hourly_load_updates = generate_updates(net,hourly_mw_data,hourly_mvar_data)
vms,vas,ps,qs,jacs = generate_phase_retrieval_data(net,hourly_load_updates)

#---------------- generate real time power flow angle data
# #--- load the real time data
# m_5mins = 12*m_hours
# inst_mvar_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mvar_data.csv"))
# inst_mw_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mw_data.csv"))

# # remove time index column
# time_idx,inst_mw_data = inst_mw_data[1:m_5mins,1],inst_mw_data[1:m_5mins,2:end]
# time_idx,inst_mvar_data = inst_mvar_data[1:m_5mins,1],inst_mvar_data[1:m_5mins,2:end]

# #--- generate the real time va data
# realtime_load_updates = generate_updates(net,inst_mw_data,inst_mvar_data)
# realtime_vas = generate_realtime_va_data(net,realtime_load_updates)


# # test the phase retrieval problem sequentially using the averaged data
hourly_phret_results = test_sequential_est_bus_voltage_phase(vms,vas,ps,qs,jacs)
    #D_f_x=Normal(0,1e-3),D_vm=Normal(0,1e-4))

# calculate the rms error
est_hourly_vas = [result["th_hat"] for result in hourly_phret_results]
rel_err_hourly_vas = [result["th_rel_err"] for result in hourly_phret_results]


# calculate the 
#hourly_rms_error = calc_hourly_rms_error(est_hourly_vas,realtime_vas)


#--- test the jacobian symmetry at the first time step
dpv1,dqv1 = Matrix(jacs[1].pv),Matrix(jacs[1].qv)
dpth1,dqth1 = Matrix(jacs[1].pth),Matrix(jacs[1].qth)
dpξ,dqξ = calc_augmented_angle_jacobians(vms[1],ps[1],qs[1],dpv1,dqv1)