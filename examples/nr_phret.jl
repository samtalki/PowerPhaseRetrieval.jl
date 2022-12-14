include("/home/sam/github/PowerSensitivities.jl/src/PowerSensitivities.jl")
include("../src/PowerPhaseRetrieval.jl")
import .PowerPhaseRetrieval as PPR
import .PowerSensitivities as PS
using JuMP,LinearAlgebra,PowerModels
using Plots,LaTeXStrings


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

function plot_estimated_angles(results)
    θ_true,θ_hat = results["th_true"],results["th_hat"]
    θrel_err = results["θrel_err"]

    
    f = plot(θ_true,label=L"$\theta_i = \theta_i^{0} + \Delta \theta_i$",marker=:circle)
    plot!(θ_hat,label=L"$\hat{\theta}_i = \theta_i^{0} + \widehat{\Delta \theta_i}$",line=:dash,marker=:square)
    xlabel!(L"Bus Index $i$")
    ylabel!(L"$\theta_i$")
    title!(
        L"$\hat{\Delta \theta_i} = -{\frac{\partial p}{\partial \theta}}^{-1} \frac{\partial p}{\partial v} \Delta v_i + {\frac{\partial p}{\partial \theta}}^{-1} \Delta p_i$  Pct Err: "*string(round(θrel_err,digits=3))
        )
    return f
end

function plot_angle_perturbations(results)
    Δθ_true,Δθ_hat_opt,Δθ_hat_deterministic = results["delta_th_true"],results["th_opt"],results["Δθ_hat"]
    f = plot(Δθ_true)
    plot!(Δθ_hat_opt)
    plot!(Δθ_hat_deterministic)
    xlabel!("Bus Index")
    return f
end


figure_path = "/home/sam/research/PowerPhaseRetrieval.jl/figures/12_14_2022/"
network_folder = "/home/sam/github/PowerSensitivities.jl/data/radial_test/"
net_names = ["case14.m","case24_ieee_rts.m","case118.m"]
paths = [network_folder*net_name for net_name in net_names]
results = Dict()
for (name,path) in zip(net_names,paths)
    net = make_basic_network(parse_file(path))
    results[name] = PPR.nr_phase_retrieval!(net)
    dpth_true,dqth_true = PS.calc_pth_jacobian(net,[1]),PS.calc_qth_jacobian(net,[1])
    dpth_hat,dqth_hat = results[name]["dpth"],results[name]["dqth"]
    angle_block_fig = plot_angle_block_comparison(dpth_true,dqth_true,dpth_hat,dqth_hat)
    angle_est_fig = plot_estimated_angles(results[name])
    #delta_angle_est_fig = plot_angle_perturbations(results[name])
    #savefig(delta_angle_est_fig,figure_path*"delta_angle_est_"*name*".pdf")
    savefig(angle_block_fig,figure_path*"phase_angle_blocks_"*name*".pdf")
    savefig(angle_est_fig,figure_path*"angle_est_"*name*".pdf")
end


