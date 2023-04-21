include("../src/PowerPhaseRetrieval.jl")
include("../src/vis/realtime.jl")
include("../src/io/pm_timeseries.jl")
import .PowerPhaseRetrieval as PPR
using PowerModels,LinearAlgebra,Plots

rel_err(x,y) = norm(x-y)/norm(y)

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


# ---------------- load case and run power flow
net = make_basic_network(parse_file("data/networks/case_RTS_GMLC.m")) #basic network
pq_buses = PPR.calc_bus_idx_of_type(net,[1])
compute_ac_pf!(net)
vm0 = abs.(calc_basic_bus_voltage(net)[pq_buses])
p0 = real.(calc_basic_bus_injection(net)[pq_buses])
q0 = imag.(calc_basic_bus_injection(net)[pq_buses])

# -----------------
# generate m_samples of 15-minute granularity timeseries data 
# ----------------
m_hours = 48 # number of hours to generate
m_15min = 4*m_hours # number of 15 minute intervals
m_5mins = 12*m_hours # number of 5 minute intervals

#---------------- ground truth: load 5-min granularity net load data 
#--- load the real time data
mvar_5m = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mvar_data.csv"))
mw_5m = DataFrame(CSV.File("data/RTS-GMLC-timeseries/nodal-dayahead/nodal_mw_data.csv"))
# remove time index column
_,mw_5m = mw_5m[1:m_5mins,1],mw_5m[1:m_5mins,2:end]
_,mvar_5m = mvar_5m[1:m_5mins,1],mvar_5m[1:m_5mins,2:end]


#---------------- Noise and delays:  generate 15 min net load data with 0.01 gaussian noise and Uni(0,2) delays for each bus.
mw_15m_gtruth,mvar_15m_gtruth = subsample_data(mw_5m),subsample_data(mvar_5m)
mw_15m_noisy_delayed,mvar_15m_noisy_delayed = generate_subsampled_data([mw_5m,mvar_5m])


#---------------- Solve power flow with the datasets
#--- generate the 5m ground truth va data
updates_5m_gtruth = generate_updates(net,mw_5m,mvar_5m)
vas_5m_gtruth = generate_realtime_va_data(net,updates_5m_gtruth)

#--- Generate the 15m ground truth va data
updates_15m_gtruth = generate_updates(net,mw_15m_gtruth,mvar_15m_gtruth)
vas_15m_gtruth = generate_realtime_va_data(net,updates_15m_gtruth)

#--- generate the 15 minute phase retrieval data
updates_15m = generate_updates(net,mw_15m_noisy_delayed,mvar_15m_noisy_delayed)
_vms,_vas,_ps,_qs,_jacs = generate_phase_retrieval_data(net,updates_15m)

#--- solve the phase retrieval problem
phret_results = test_sequential_est_bus_voltage_phase(_vms,_vas,_ps,_qs,_jacs)#,D_f_x=Normal(0,1e-3),D_vm=Normal(0,1e-4))

# calculate the rms error
vas_15m_est = [result["th_hat"] for result in phret_results]
rel_err_15m = [rel_err(va_hat_t,va_t) for (va_hat_t,va_t) in zip(vas_15m_est,vas_15m_gtruth)]

# plot the 15minute ground truth, estimated ground truth, and 5 minute ground truth


# plot the results
for bus_idx in 1:length(pq_buses)
    θ_5min_gtruth =  [e_t[bus_idx] for e_t in vas_5m_gtruth]
    θ_15min_gtruth = [] #ground truth averages angles at each hour repeatedly applied over the 5min granularity time scale
    θ_15min_est = [] #real time averaged angles applied repeatedly over the 5min granularity time scale
    for k = 1:m_15min
        for t =1:3
            push!(θ_15min_gtruth,vas_15m_gtruth[k][bus_idx])
            push!(θ_15min_est,vas_15m_est[k][bus_idx])
        end
    end

    #---- plot with relative real time error
    rel_err_5m = round(norm(θ_15min_est - θ_5min_gtruth)/norm(θ_5min_gtruth)*100,digits=3)
    p = plot_avg_vs_inst_tseries(θ_15min_est,θ_15min_gtruth,θ_5min_gtruth)
    title!(p,"Bus $(bus_idx), " * L"$p=0.1$, $\sigma_{\mathrm{mes}} = 0.01$, " * "5->15 min rel. err. $(rel_err_5m) %")
    savefig(p,"figures/sampling-rate-and-noise/avg_vs_inst_$(bus_idx).pdf")
end
