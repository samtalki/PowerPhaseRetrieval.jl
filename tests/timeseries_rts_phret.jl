# create synthetic data for one day at 15 minute granularity driven by smart meter data varying demands. 
# Then assign random delays within a 15 minute range (e.g. uniform distribution). See the impact of lack of synchronization in the data stream. 
# My strong believe is that the method will be robust enough to generate the matrices and phase retrieval that have small error. 

include("../src/PowerPhaseRetrieval.jl")
include("../src/vis/realtime.jl")
include("../src/io/pm_timeseries.jl")
import .PowerPhaseRetrieval as PPR
using PowerModels,LinearAlgebra,Plots

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
function calc_hourly_mean_relative_error(hourly_phret_results::Vector,realtime_phase_angles::Vector)
    @assert length(realtime_phase_angles) == 12*length(hourly_phret_results)
    real_time_rel_errs = []
    for hr=1:length(hourly_phret_results)

        #get the forecasted θ for this hour
        θ_hat_hr = hourly_phret_results[hr] 

        #get all of the real time θs for this hour
        θ_hr = realtime_phase_angles[12*(hr-1)+1:12*hr]
        @assert length(θ_hr) == 12

        #calculate the individual relative error for each 5 minute interval
        for i=1:12
            push!(real_time_rel_errs,norm(θ_hat_hr - θ_hr[i])/norm(θ_hr[i]))
        end
    end
    return real_time_rel_errs
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
#--- load the real time data
m_5mins = 12*m_hours
inst_mvar_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mvar_data.csv"))
inst_mw_data = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mw_data.csv"))

# remove time index column
time_idx,inst_mw_data = inst_mw_data[1:m_5mins,1],inst_mw_data[1:m_5mins,2:end]
time_idx,inst_mvar_data = inst_mvar_data[1:m_5mins,1],inst_mvar_data[1:m_5mins,2:end]

#--- generate the real time va data
realtime_load_updates = generate_updates(net,inst_mw_data,inst_mvar_data)
realtime_vas = generate_realtime_va_data(net,realtime_load_updates)


# # test the phase retrieval problem sequentially using the averaged data
hourly_phret_results = test_sequential_est_bus_voltage_phase(vms,vas,ps,qs,jacs)#,D_f_x=Normal(0,1e-3),D_vm=Normal(0,1e-4))

# calculate the rms error
est_hourly_vas = [result["th_hat"] for result in hourly_phret_results]
rel_err_hourly_vas = [result["th_rel_err"] for result in hourly_phret_results]


# calculate the 
real_time_rel_errs = calc_hourly_mean_relative_error(est_hourly_vas,realtime_vas)

# plot the results
for bus_idx = 1:5
    θ_inst_5min =  [e_t[bus_idx] for e_t in realtime_vas]
    θ_hat_inst_hr = [] #averaged estimates for the angles at each hour repeatedly applied over the 5min granularity time scale
    θ_inst_hr = [] #real time averaged angles applied repeatedly over the 5min granularity time scale
    for h = 1:m_hours
        for t =1:12
            push!(θ_hat_inst_hr,est_hourly_vas[h][bus_idx])
            push!(θ_inst_hr,vas[h][bus_idx])
        end
    end
    p = plot_avg_vs_inst_tseries(θ_hat_inst_hr,θ_inst_hr,θ_inst_5min)
    savefig(p,"figures/avg_vs_inst_$(bus_idx).pdf")
end



# #--- test the jacobian symmetry at the first time step
# dpv1,dqv1 = Matrix(jacs[1].pv),Matrix(jacs[1].qv)
# dpth1,dqth1 = Matrix(jacs[1].pth),Matrix(jacs[1].qth)
# dpξ,dqξ = calc_augmented_angle_jacobians(vms[1],ps[1],qs[1],dpv1,dqv1)

# #--- calculate the errors

