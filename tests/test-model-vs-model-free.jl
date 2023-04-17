# --------
# Test the phase retrieval problem with a known and unknown topology
# --------

include("../src/PowerPhaseRetrieval.jl")
include("../src/core/rectangular_sensitivities.jl")
import .PowerPhaseRetrieval as PPR
using LinearAlgebra,PowerModels
using Statistics,Random
using Plots,LaTeXStrings,ColorSchemes

# ---------------- load case and run power flow
net = make_basic_network(parse_file("data/networks/case_RTS_GMLC.m")) #basic network
pq_buses = PPR.calc_bus_idx_of_type(net,[1])
compute_ac_pf!(net)

# ----------------- get the initial values
v0 = calc_basic_bus_voltage(net)[pq_buses]
vm0 = abs.(calc_basic_bus_voltage(net)[pq_buses])
p0 = real.(calc_basic_bus_injection(net)[pq_buses])
q0 = imag.(calc_basic_bus_injection(net)[pq_buses])

# ----------------- get the topology data
Y = calc_basic_admittance_matrix(net)[pq_buses,pq_buses]
G,B = real.(Y),imag.(Y)

# ----------------- get the sensitivities
∂pe,∂qe,∂pf,∂qf,∂v²e,∂v²f = calc_rectangular_jacobian_sensitivities(G,B,v0)
J = [
    ∂pe  ∂pf;
    ∂qe  ∂qf;
    ∂v²e  ∂v²f
]
n3,n2 = size(J)
n = length(v0)


# ----------------- get the measurements
e0,f0 = real.(v0),imag.(v0) # real and imaginary part of the voltage
x_true = [e0;f0]
y_true = J*x_true



#---------------------- Create and solve the model and plot results
#--- Global plotting parameters

figure_path = "figures/with-without-topology/"
network_folder = "data/networks/"
network_names = ["case24_ieee_rts","case_RTS_GMLC"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]
color_scheme = color_list(:tab10)


#--- Global noise parameters
sigma_jac = [0.05,0.1]
sigma_noise = [0.01,0.05]


#---- for each jacobian noise level, plot the phase angle noise levels 
#Plot the phase estimates for all sigma noise levels
for (case_name,path) in zip(network_names,network_paths)
    net = make_basic_network(parse_file(path))
    
    
    plots_by_jac_noise = []
    for s_J in sigma_jac #for each jacobian noise level
        sigma_model_free_results = []
        model_based_results = []
        for s in sigma_noise
            s_model_free = PPR.est_stochastic_bus_voltage_phase!(net,sigma_noise=s,sigma_jac=s_J)
            push!(sigma_model_free_results,s_model_free)
            
            # --- model based
            s_model_based = PPR.solve_rectangular_phret!(net,σ=0,μ=0) # ---- no noise for the model-based case
            push!(model_based_results,s_model_based)
        
        end
        case_name = sigma_model_free_results[1]["case_name"]
       
        #Plot the ground truth angle
        θ_true = sigma_model_free_results[1]["th_true"]
        angle_fig = plot(
            θ_true,
            label=nothing,#label=L"$\theta_i$ true",
            marker=:circle,
            color=:black,
            line=:solid,
            lw=2.5,
            ms=3.25,
            alpha=0.85,
            legend=:best,
            size=(800,(400/600)*800),
            ylabel=L"$\theta$ (radians)",
            legendcol=2
        )


        for (idx_res,(mf_result,mb_result)) in enumerate(zip(sigma_model_free_results,model_based_results))
            
            #---- plot the model-free mf_result
            mf_θ_hat = mf_result["th_hat"]
            th_sq_errs = mf_result["th_sq_errs"]
            th_rel_err = round(mf_result["th_rel_err"],digits=3)
            sigma = mf_result["sigma_noise"]
            yerr = 2*sqrt.(th_sq_errs)
            plot!(
                mf_θ_hat,
                label=L"MF $\sigma =$"*string(sigma)*", rel. err.="*string(th_rel_err)*"%",
                line=:dot,
                marker=:square,
                ribbon=yerr,
                ms=2.5,
                lw=1.5,
                fillalpha=0.15,
                alpha=0.6,
                titlefontsize=13,
                legendfontsize=7,
                color=color_scheme[idx_res]
                #left_margin=1.2
            )

            if idx_res == 1 # ---- plot the model based result on only the first iteration
                # ---- plot the model based results
                n = length(mf_θ_hat)
                v_hat = mb_result["x_hat"][1:n] + im*mb_result["x_hat"][n+1:end]
                mb_θ_hat = angle.(v_hat)
                th_rel_err = round(mb_result["err"]*100,digits=3) # relative error in percent
                plot!(
                    mb_θ_hat,
                    label=L"MB, rel. err.="*string(th_rel_err)*"%",
                    line=:dash,
                    marker=:diamond,
                    ms=2.5,
                    lw=1.5,
                    fillalpha=0.15,
                    alpha=0.6,
                    titlefontsize=13,
                    legendfontsize=7,
                    color=color_scheme[9],
                    #left_margin=1.2
                )
            end
        end
        dpv_err,dqv_err = round(sigma_model_free_results[1]["obs_dpv_rel_err"],digits=3),round(sigma_model_free_results[1]["obs_dqv_rel_err"],digits=3)
        title!(string(case_name)*": "*L"$\hat{\theta}$ with Jac. rel. err.="*string(dpv_err)*"%")
        push!(plots_by_jac_noise,angle_fig)
    end
    p_tot = plot(plots_by_jac_noise...,layout=(length(plots_by_jac_noise),1))
    xlabel!(L"Bus Index $i$")
    
    savefig(p_tot,figure_path*"_mb_v_mf_noisy_"*string(case_name)*".pdf")
end



# --- plot the model based results
