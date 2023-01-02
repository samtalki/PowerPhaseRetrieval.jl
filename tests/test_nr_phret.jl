include("/home/sam/github/PowerSensitivities.jl/src/PowerSensitivities.jl")
include("/home/sam/Research/PowerPhaseRetrieval/src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import .PowerSensitivities as PS
using JuMP,LinearAlgebra,PowerModels
using Plots,LaTeXStrings
theme(:ggplot2)

#--- Global plotting parameters
angle_block_figure_path = "/home/sam/Research/PowerPhaseRetrieval/figures/spring_23/est_phase_jacobians/"
angle_est_figure_path = "/home/sam/Research/PowerPhaseRetrieval/figures/spring_23/est_bus_voltage_phase/"
phasor_figure_path = "/home/sam/Research/PowerPhaseRetrieval/figures/spring_23/est_bus_phasors/"
network_folder = "/home/sam/github/PowerSensitivities.jl/data/radial_test/"
network_names = ["case14","case24_ieee_rts","case_ieee30","case_RTS_GMLC","case118"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]


function plot_angle_block_comparison(dpth_true,dqth_true,dpth_hat,dqth_hat)
    
    dpth_err = norm(dpth_true-dpth_hat)/norm(dpth_true)*100
    dqth_err = norm(dqth_true-dqth_hat)/norm(dqth_true)*100
    
    #--- Make plots
    hp = heatmap(
        dpth_true,
        title=L"$\frac{\partial p}{\partial \theta}$ (At power flow sol.)",
        c=:curl,
        ylabel="Bus index",
        )
    hq = heatmap(
        dqth_true,
        title=L"$\frac{\partial q}{\partial \theta}$ (At power flow sol.)",
        c=:curl,
        
        )
    hat_hp = heatmap(
        dpth_hat,
        title=L"Est. $\frac{\partial p}{\partial \theta}$ Rel. Error:"*string(round(dpth_err,digits=3))*"%",
        c=:curl,
        xlabel="Bus index",
        ylabel="Bus index",
        )
    hat_hq = heatmap(
        dqth_hat,
        title=L"Est. $\frac{\partial q}{\partial \theta}$ Rel. Error:"*string(round(dqth_err,digits=3))*"%",
        c=:curl,
        xlabel="Bus index",
    )
    fig = plot(
        hp,hq,hat_hp,hat_hq,
        titlefontsize=11,
        tickfontsize=6,
        labelfontsize=10,
        grid=true,
        cb=:best
    )
    return fig
end

function plot_estimated_angles(results::Dict)
    case_name = results["case_name"]
    θ_true,θ_hat = results["th_true"],results["th_hat"]
    θrel_err = results["th_rel_err"]
    th_sq_errs = results["th_sq_errs"]
    yerr = 2*sqrt.(th_sq_errs)
    f = plot(
        θ_true,
        label=L"$\theta_i$",
        marker=:circle)
    plot!(
        θ_hat,
        label=L"$\hat{\theta}_i$",
        line=:dash,
        marker=:square,
        ribbon=yerr    
    )
    xlabel!(L"Bus Index $i$")
    ylabel!(L"$\theta_i$")
    title!(string(case_name)*L" $\hat{\theta}$ Rel. Error: "*string(round(θrel_err,digits=3))*"%",fontsize=11)
    # title!(
    #     L"$\hat{\Delta \theta_i} = -{\frac{\partial p}{\partial \theta}}^{-1} \frac{\partial p}{\partial v} \Delta v_i + {\frac{\partial p}{\partial \theta}}^{-1} \Delta p_i$  Pct Err: "*string(round(θrel_err,digits=3))
    #     )
    return f
end

"""
Given an array of sigma noise levels and a network, plot all estimated phase angles by noise level.
"""
function plot_estimated_angles(network::Dict{String,Any},sigma_noise::AbstractArray{Float64})
    sigma_results = []
    for s in sigma_noise
        push!(sigma_results,PPR.est_bus_voltage_phase!(network,sigma_noise=s))
    end
    case_name = sigma_results[1]["case_name"]
    #Plot th_true
    θ_true = sigma_results[1]["th_true"]
    angle_fig = plot(
        θ_true,
        label=L"$\theta_i$ true",
        marker=:circle,
        color=:black,
        line=:solid,
        lw=2.5,
        ms=3.25,
        alpha=0.85,
        legend=:bottomleft
    )

    for results in sigma_results
        θ_hat = results["th_hat"]
        th_sq_errs = results["th_sq_errs"]
        sigma = results["sigma_noise"]
        yerr = 2*sqrt.(th_sq_errs)
        plot!(
            θ_hat,
            label=L"$\hat{\theta}_i$ $\sigma =$"*string(sigma*100)*"%",
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
    ylabel!(L"Voltage phase angle $\theta_i$ (radians)")
    title!(L"$\hat{\theta}$ by noise level "*string(case_name))
    savefig(angle_fig,angle_est_figure_path*"angle_est_"*string(case_name)*".pdf")
    return p
end


"""
Plot estimated sinsuoids
"""
function plot_estimated_sinusoids(results::Dict;t=1:0.01:10,f=60)
    case_name = results["case_name"]
    θ_true,θ_hat = results["th_true"],results["th_hat"]
    v_rect_true,v_rect_hat = results["v_rect_true"],results["v_rect_hat"]
    vm_true,vm_hat = abs.(v_rect_true),abs.(vm_hat)
    n = length(vm_true)
    #Make sinusoid matrices
    ac_true = zeros(length(t),n)
    ac_est = zeros(length(t),n)

    for i=1:n
        ac_true[:,i] = vm_true[i]*cos.(2π*f*t .+ θ_true[i])
        ac_est[:,i] = vm_hat[i]*cos.(2π*f*t .+ θ_hat[i])
    end

    #bus index labels
    label = [string(i) for i=1:n]

    p1= plot(t,
        ac_est,
        label=label,
        legendtitle="Bus")
    plot!(t,
        ac_true,
        legend=false,
        line=:dash 
    )
    xlabel!(L"Time $t$ (s)")
    ylabel!(L"Sinusoid")
    return p
end


"""
Given a single sigma_noise level, plot all of the estimation figures.
"""
function plot_phase_estimates(sigma_noise::Real;sel_bus_types=[1])
    results = Dict()
    for (name,path) in zip(network_names,network_paths)
        net = make_basic_network(parse_file(path))
        results[name] = PPR.est_bus_voltage_phase!(net,sel_bus_types=sel_bus_types,sigma_noise=sigma_noise)
        
        #Block angle block estimates
        dpth_true,dqth_true = PS.calc_pth_jacobian(net,sel_bus_types),PS.calc_qth_jacobian(net,sel_bus_types)
        dpth_hat,dqth_hat = results[name]["dpth"],results[name]["dqth"]
        angle_block_fig = plot_angle_block_comparison(dpth_true,dqth_true,dpth_hat,dqth_hat)
        
        #Plot estimated angles
        angle_fig = plot_estimated_angles(results[name])
        
        #Plot estimated voltage phasors
        #phasor_fig = plot_estimated_phasors(results[name])

        #Save figures
        savefig(angle_block_fig,angle_block_figure_path*"sigma_"*string(sigma_noise)*"_phase_angle_blocks_"*name*".pdf")
        savefig(angle_fig,angle_est_figure_path*"sigma_"*string(sigma_noise)*"_angle_est_"*name*".pdf")
        #savefig(phasor_fig,phasor_figure_path*"phasor_est_"*name*".pdf")
    end
end



#-------------- Begin plotting
sigma_noise = [0.1,0.25,0.50]

#Plot the phase estimates for all sigma noise levels
for (name,path) in zip(network_names,network_paths)
    net = make_basic_network(parse_file(path))
    plot_estimated_angles(net,sigma_noise)
end

#Plot the individual phase and jacobian estimates for all noise level
for s in sigma_noise
    plot_phase_estimates(s)
end



#-----------------------------------------


# #TODO: Remove or Cleanup ---- Ineffective plotting functions
# """
# Given a results dict plot the estimated phasors in polar coordinates.
# """
# function plot_estimated_phasors(results::Dict)
#     v_rect_true,v_rect_hat = results["v_rect_true"],results["v_rect_hat"]
#     r_true,r_hat = abs.(v_rect_true),abs.(v_rect_hat)
#     th_hat,th_true = results["th_hat"],results["th_true"]
#     v_rect_rel_err = results["v_rect_rel_err"]
#     yerr = abs.(v_rect_true .- v_rect_hat)
#     # PLJS.plot(
#     #     PLJS.scatterpolar(r=r_true,theta=th_true,mode="markerss"),
#     #     PLJS.Layout(polar=attr(angularaxis_direction="counterclockwise", sector=(0, 90)))

#     # )
#     p = plot(
#         th_true,
#         r_true,
#         proj=:polar,
#         m=2,
#         marker=:circle,
#         line=:dash,
#         label=L"v_i(\cos(\theta_i) + j \sin(\theta_i))",
#         xlims=[minimum(th_true)-0.1,maximum(th_true)+0.1],
#         yerr=yerr
#     )
#     plot!(
#         th_hat,r_hat,
#         proj=:polar,
#         m=2,
#         marker=:circle,
#         line=:dash,
#         label=L"v_i(\cos(\hat{\theta}_i) + j \sin(\hat{\theta}_i))"
#         #ribbon=yerr
#     )
    
#     title!(
#         L"Est. Bus Voltage Phasors Rel. Error: "*string(round(v_rect_rel_err,digits=3))*"%",
#         fontsize=11
#     )
#     return p
# end



# function plot_angle_perturbations(results)
#     Δθ_true,Δθ_hat_opt,Δθ_hat_deterministic = results["delta_th_true"],results["th_opt"],results["Δθ_hat"]
#     f = plot(Δθ_true)
#     plot!(Δθ_hat_opt)
#     plot!(Δθ_hat_deterministic)
#     xlabel!("Bus Index")
#     return f
# end