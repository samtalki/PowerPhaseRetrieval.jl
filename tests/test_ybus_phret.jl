include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM
using JuMP,SCS,LinearAlgebra
using ColorSchemes
using Plots,LaTeXStrings
theme(:ggplot2)

network_folder = "data/"
network_names = ["case14","case24_ieee_rts","case_ieee30","case_RTS_GMLC","case118"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]
sigma_noise = [0.01,0.025,0.05]

#--- figure paths
Vangle_est_figure_path = "figures/spring_23/ybus/est_bus_voltage_phase/"
Iangle_est_figure_path = "figures/spring_23/ybus/est_bus_current_phase/"

"""
Plot estimated voltage phases
"""
function plot_estimated_voltage_angles(solutions::Vector)
    case_name = solutions[1].data.case_name
    θ_true = solutions[1].data.Vangle
    Vangle_fig = plot(
        θ_true,
        label=L"$\theta_i$ true",
        marker=:circle,
        color=:black,
        line=:solid,
        lw=2.5,
        ms=3.25,
        alpha=0.85,
        legend=:best
    )
    for sol in solutions
        θ_hat = sol.Vangle
        th_sq_errs = sol.Vangle_sq_err
        th_rel_err = round(sol.Vangle_rel_err,digits=3)
        sigma = sol.data.sigma_noise
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
        )
    end
    xlabel!(L"Bus Index $i$")
    ylabel!(L"Voltage phase angles $\theta_i$ (radians)")
    title!(string(case_name)*L": $\hat{\theta}$ by meas. noise level")
    savefig(Vangle_fig,Vangle_est_figure_path*"Vangle_est_"*string(case_name)*".pdf")
    return Vangle_fig
end

"""
Plot estimated current phases
"""
function plot_estimated_current_angles(solutions::Vector)
    case_name = solutions[1].data.case_name
    θ_true = solutions[1].data.Iangle
    Iangle_fig = plot(
        θ_true,
        label=L"$\phi_i$ true",
        marker=:circle,
        color=:black,
        line=:solid,
        lw=2.5,
        ms=3.25,
        alpha=0.85,
        legend=:best
    )
    for sol in solutions
        θ_hat = sol.Iangle
        th_sq_errs = sol.Iangle_sq_err
        th_rel_err = round(sol.Iangle_rel_err,digits=3)
        sigma = sol.data.sigma_noise
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
        )
    end
    xlabel!(L"Bus Index $i$")
    ylabel!(L"Current phase angles $\phi_i$ (radians)")
    title!(string(case_name)*L": $\hat{\phi}$ by meas. noise level")
    savefig(Iangle_fig,Iangle_est_figure_path*"Iangle_est_"*string(case_name)*".pdf")
    return Iangle_fig
end

sols_by_case = Dict()
for (case_name,path) in zip(network_names,network_paths)    
    net = PM.make_basic_network(PM.parse_file(path))
    solutions = PPR.solve_ybus_phasecut!(net,sigma_noise)
    sols_by_case[case_name] = solutions
    plot_estimated_current_angles(solutions)
    plot_estimated_voltage_angles(solutions)
end

for (n,s) in sols_by_case
    println("Case ",n)
    println("Irect_rel_err: "*string([s.Irect_rel_err for s in s]))
    println("Vrect_rel_err: "*string([s.Vrect_rel_err for s in s]))
    println("Iangle_rel_err: "*string([s.Iangle_rel_err for s in s]))
    println("Vangle_rel_err: "*string([s.Vangle_rel_err for s in s]))
end


