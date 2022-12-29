include("/home/sam/github/PowerSensitivities.jl/src/PowerSensitivities.jl")
include("/home/sam/Research/PowerPhaseRetrieval/src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import .PowerSensitivities as PS
using JuMP,LinearAlgebra,PowerModels
using Plots,LaTeXStrings
theme(:ggplot2)


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



angle_block_figure_path = "/home/sam/Research/PowerPhaseRetrieval/figures/spring_23/est_phase_jacobians/"
angle_est_figure_path = "/home/sam/Research/PowerPhaseRetrieval/figures/spring_23/est_bus_voltage_phase/"
phasor_figure_path = "/home/sam/Research/PowerPhaseRetrieval/figures/spring_23/est_bus_phasors/"
network_folder = "/home/sam/github/PowerSensitivities.jl/data/radial_test/"
net_names = ["case14","case24_ieee_rts","case_ieee30","case_RTS_GMLC","case118"]
paths = [network_folder*net_name*".m" for net_name in net_names]
results = Dict()
for (name,path) in zip(net_names,paths)
    net = make_basic_network(parse_file(path))
    results[name] = PPR.est_bus_voltage_phase!(net)
    
    #Block angle block estimates
    dpth_true,dqth_true = PS.calc_pth_jacobian(net,[1]),PS.calc_qth_jacobian(net,[1])
    dpth_hat,dqth_hat = results[name]["dpth"],results[name]["dqth"]
    angle_block_fig = plot_angle_block_comparison(dpth_true,dqth_true,dpth_hat,dqth_hat)
    
    #Plot estimated angles
    angle_fig = plot_estimated_angles(results[name])
    
    #Plot estimated voltage phasors
    #phasor_fig = plot_estimated_phasors(results[name])

    #Save figures
    savefig(angle_block_fig,angle_block_figure_path*"phase_angle_blocks_"*name*".pdf")
    savefig(angle_fig,angle_est_figure_path*"angle_est_"*name*".pdf")
    #savefig(phasor_fig,phasor_figure_path*"phasor_est_"*name*".pdf")
end



#TODO: Remove or Cleanup ---- Ineffective plotting functions
"""
Given a results dict plot the estimated phasors in polar coordinates.
"""
function plot_estimated_phasors(results::Dict)
    v_rect_true,v_rect_hat = results["v_rect_true"],results["v_rect_hat"]
    r_true,r_hat = abs.(v_rect_true),abs.(v_rect_hat)
    th_hat,th_true = results["th_hat"],results["th_true"]
    v_rect_rel_err = results["v_rect_rel_err"]
    yerr = abs.(v_rect_true .- v_rect_hat)
    # PLJS.plot(
    #     PLJS.scatterpolar(r=r_true,theta=th_true,mode="markerss"),
    #     PLJS.Layout(polar=attr(angularaxis_direction="counterclockwise", sector=(0, 90)))

    # )
    p = plot(
        th_true,
        r_true,
        proj=:polar,
        m=2,
        marker=:circle,
        line=:dash,
        label=L"v_i(\cos(\theta_i) + j \sin(\theta_i))",
        xlims=[minimum(th_true)-0.1,maximum(th_true)+0.1],
        yerr=yerr
    )
    plot!(
        th_hat,r_hat,
        proj=:polar,
        m=2,
        marker=:circle,
        line=:dash,
        label=L"v_i(\cos(\hat{\theta}_i) + j \sin(\hat{\theta}_i))"
        #ribbon=yerr
    )
    
    title!(
        L"Est. Bus Voltage Phasors Rel. Error: "*string(round(v_rect_rel_err,digits=3))*"%",
        fontsize=11
    )
    return p
end



function plot_angle_perturbations(results)
    Δθ_true,Δθ_hat_opt,Δθ_hat_deterministic = results["delta_th_true"],results["th_opt"],results["Δθ_hat"]
    f = plot(Δθ_true)
    plot!(Δθ_hat_opt)
    plot!(Δθ_hat_deterministic)
    xlabel!("Bus Index")
    return f
end