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
angle_block_figure_path = "figures/spring_23/date_01032023/est_phase_jacobians/"
angle_est_figure_path = "figures/spring_23/date_01032023/est_bus_voltage_phase/"
phasor_figure_path = "figures/spring_23/date_01032023/est_bus_phasors/"
sinusoid_figure_path = "figures/spring_23/date_01032023/est_sinusoids/"
network_folder = "data/"
network_names = ["case14","case24_ieee_rts","case_ieee30","case_RTS_GMLC","case89pegase","case118"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]

#--- GLOBAL NOISE parameters

sigma_noise = [0.1,0.2,0.5]

function plot_angle_block_comparison(case_name::String,dpth_true,dqth_true,dpth_hat,dqth_hat)
    
    dpth_err = norm(dpth_true-dpth_hat)/norm(dpth_true)*100
    dqth_err = norm(dqth_true-dqth_hat)/norm(dqth_true)*100
    
    #--- Make plots
    hp = heatmap(
        dpth_true,
        title=case_name*L" $\partial p / \partial \theta$ (at NR sol.)",
        c=:curl,
        ylabel="Bus index",
        titlefontsize=12,
        )
    hq = heatmap(
        dqth_true,
        title=case_name*L" $\partial q / \partial \theta$ (at NR sol.)",
        c=:curl,
        titlefontsize=12,
        )
    hat_hp = heatmap(
        dpth_hat,
        title=L"est. $\partial p / \partial \theta$ (rel. err.:"*string(round(dpth_err,digits=3))*"%)",
        c=:curl,
        xlabel=L"Bus index $i$",
        ylabel=L"Bus index $i$",
        titlefontsize=12,
        )
    hat_hq = heatmap(
        dqth_hat,
        title=L"est. $\partial q / \partial \theta$ (rel. err.: "*string(round(dqth_err,digits=3))*"%)",
        c=:curl,
        titlefontsize=12,
        xlabel=L"Bus index $i$",
    )
    err_hp = heatmap(
        abs.(dpth_true - dpth_hat),
        title=L"$\partial p / \partial \theta$ elementwise abs. err.",
        c=:curl,
        titlefontsize=12,
        xlabel=L"Bus index $i$",
        ylabel=L"Bus index $i$"
    )
    err_hq = heatmap(
        abs.(dqth_true - dqth_hat),
        title=L"$\partial q / \partial \theta$ elementwise abs. err.",
        titlefontsize=12,
        c=:curl,
        xlabel=L"Bus index $i$"
    )

    fig = plot(
        hp,hq,hat_hp,hat_hq,#err_hp,err_hq,
        #layout=(2,2),
        titlefontsize=11,
        grid=false,
        tickfontsize=6,
        labelfontsize=10,
        #plot_title=case_name,
       # plot_title_fontsize=13,
       # plot_title_vspace=0.15,
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
        legend=:best
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
            
        )
    end
    xlabel!(L"Bus Index $i$")
    ylabel!(L"Voltage phase angles $\theta_i$ (radians)")
    title!(string(case_name)*L": $\hat{\theta}$ by meas. noise level")
    savefig(angle_fig,angle_est_figure_path*"angle_est_"*string(case_name)*".pdf")
    return angle_fig
end


"""
Plot estimated sinsuoids
"""
function plot_estimated_sinusoids(results::Dict;t=0:1e-5:1.5/60,f=60,num_bus=5)
    case_name = results["case_name"]
    θ_true,θ_hat = results["th_true"],results["th_hat"]
    v_rect_true,v_rect_hat = results["v_rect_true"],results["v_rect_hat"]
    vm_true,vm_hat = abs.(v_rect_true),abs.(v_rect_hat)
    
    #skip the slack and find the most different elements
    θ_distinction = [abs(θ_i - mean(θ_true)) for θ_i in θ_true]
    sel_bus = []
    for ix=1:num_bus
        mxval,mxix = findmax(θ_distinction)
        θ_distinction[mxix] = 0
        push!(sel_bus,mxix)
    end

    #Make sinusoid matrices
    ac_true = zeros(length(t),num_bus)
    ac_est = zeros(length(t),num_bus)

    for (i,bus_idx) in enumerate(sel_bus)
        ac_true[:,i] = vm_true[bus_idx]*cos.(2π*f*t .+ θ_true[bus_idx])
        ac_est[:,i] = vm_hat[bus_idx]*cos.(2π*f*t .+ θ_hat[bus_idx])
    end

    p1 = plot(t,
        ac_est,
        palette=:default,
        line=:dash,
        ylim=[-1.75,1.75],
        label = permutedims([string(idx) for idx in sel_bus]),
        legend= :bottom,
        legendtitle=L"Bus idx. $i$",
        legendfontsize=5,
        legendtitlefontsize=5,
        legend_column=num_bus,
        title=L"est. $\hat{v}_i(t) = |\bar{v}_i|\cos(2 \pi f t - \hat{\theta}_i)$",
        ylabel="AC voltage"
    )
    p2 = plot(t,
        ac_true,
        palette=:default,
        title = L"actual $v(t) = |\bar{v}_i|\cos(2\pi f t - \theta_i)$",
        ylim=[-1.75,1.75],
        label = permutedims([string(idx) for idx in sel_bus]),
        legend= :bottom,
        legendtitle=L"Bus idx. $i$",
        legendfontsize=5,
        legendtitlefontsize=5,
        legend_column=num_bus,
        xlabel=L"Time $t$ (s)",
        ylabel="AC voltage"
        
    )
    # rmse(x_true,x_hat) = sqrt((1/length(x_true))*sum([abs(x_true_i - x_hat_i)^2 for (x_true_i,x_hat_i) in zip(x_true,x_hat)]))
    # p3 = plot([i for i=1:length(vm_true)],
    #     [rmse(ac_true_i,ac_est_i) for (ac_true_i,ac_est_i) in zip(ac_true,ac_est)],#abs.(ac_true .- ac_est),
    #     xlabel=L"Bus index $i=1,\dots,n$",
    #     ylabel=L"RMSE",
    #     line=:stem
    # )
    p = plot(p1,p2,layout=(2,1))
    
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
        #Plot estimated blocks
        angle_block_fig = plot_angle_block_comparison(name,dpth_true,dqth_true,dpth_hat,dqth_hat)
        #Plot estimated angles
        angle_fig = plot_estimated_angles(results[name])    
        #Plot estimated sinusoids
        sinusoid_fig = plot_estimated_sinusoids(results[name])
    
        #Save figures
        savefig(angle_block_fig,angle_block_figure_path*"sigma_"*string(sigma_noise)*"_phase_angle_blocks_"*name*".pdf")
        savefig(angle_fig,angle_est_figure_path*"sigma_"*string(sigma_noise)*"_angle_est_"*name*".pdf")
        savefig(sinusoid_fig,sinusoid_figure_path*"sigma_"*string(sigma_noise)*"_sinus_est_"*name*".pdf")
        #savefig(phasor_fig,phasor_figure_path*"phasor_est_"*name*".pdf")
    end
end




#-------------- Begin plotting

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