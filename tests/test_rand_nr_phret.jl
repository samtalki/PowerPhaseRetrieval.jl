#include("test_nr_phret.jl")
include("/home/sam/github/PowerSensitivities.jl/src/PowerSensitivities.jl")
include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import .PowerSensitivities as PS
using JuMP,LinearAlgebra,PowerModels
using ColorSchemes
using Plots,LaTeXStrings
using Statistics,Random
theme(:ggplot2)

#--- Global plotting parameters

angle_est_figure_path = "figures/spring_23/date_01032023/noisy_jac_est_bus_voltage_phase/"
sinusoid_figure_path = "figures/spring_23/date_01032023/est_sinusoids/"
network_folder = "data/"
network_names = ["case14","case24_ieee_rts","case_ieee30","case_RTS_GMLC","case89pegase","case118"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]

#--- Global noise parameters
sigma_jac = [0.05,0.1]
sigma_noise = [0.01,0.05,0.15]

"""
Given:
- an array of sigma noise levels, 
- an array of jacobian noise levels, 
- and a network, 
"""

#---- for each jacobian noise level, plot the phase angle noise levels 
#Plot the phase estimates for all sigma noise levels
for (case_name,path) in zip(network_names,network_paths)
    net = make_basic_network(parse_file(path))
    plots_by_jac_noise = []

    for s_J in sigma_jac #for each jacobian noise level

        sigma_results = []
        for s in sigma_noise
            s_results = PPR.est_stochastic_bus_voltage_phase!(net,sigma_noise=s,sigma_jac=s_J)
            push!(sigma_results,s_results)
        end
        case_name = sigma_results[1]["case_name"]
        #Plot th_true
        θ_true = sigma_results[1]["th_true"]
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
            ylabel=L"$\theta$ (radians)"
        )

        for results in sigma_results
            θ_hat = results["th_hat"]
            th_sq_errs = results["th_sq_errs"]
            th_rel_err = round(results["th_rel_err"],digits=3)
            sigma = results["sigma_noise"]
            yerr = 2*sqrt.(th_sq_errs)
            plot!(
                θ_hat,
                label=L"$\sigma =$"*string(sigma)*", rel. err.="*string(th_rel_err)*"%",
                line=:dot,
                marker=:square,
                ribbon=yerr,
                ms=2.5,
                lw=1.5,
                fillalpha=0.15,
                alpha=0.6,
                titlefontsize=13,
                legendfontsize=7,
                #left_margin=1.2
            )
        end
        dpv_err,dqv_err = round(sigma_results[1]["obs_dpv_rel_err"],digits=3),round(sigma_results[1]["obs_dqv_rel_err"],digits=3)
        title!(string(case_name)*": "*L"$\hat{\theta}$ with Jac. rel. err.="*string(dpv_err)*"%")
        push!(plots_by_jac_noise,angle_fig)
    end
    p_tot = plot(plots_by_jac_noise...,layout=(length(plots_by_jac_noise),1))
    xlabel!(L"Bus Index $i$")
    
    savefig(p_tot,angle_est_figure_path*"noisy_jac_angle_est_"*string(case_name)*".pdf")
end