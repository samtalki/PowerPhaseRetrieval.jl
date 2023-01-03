include("test_nr_phret.jl")

#--- Global plotting parameters

angle_est_figure_path = "figures/spring_23/date_01032023/noisy_jac_est_bus_voltage_phase/"
sinusoid_figure_path = "figures/spring_23/date_01032023/est_sinusoids/"
network_folder = "/home/sam/github/PowerSensitivities.jl/data/radial_test/"
network_names = ["case14","case24_ieee_rts","case_ieee30","case_RTS_GMLC","case118"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]

#--- Global noise parameters
sigma_jac = [0.01,0.05,0.15]
sigma_noise = [0.1,0.25,0.50]

"""
Given:
- an array of sigma noise levels, 
- an array of jacobian noise levels, 
- and a network, 

plot all estimated phase angles by noise level. with subplots by jac noise level.
"""
function plot_estimated_angles(network::Dict{String,Any},sigma_noise::AbstractArray{Float64},sigma_jac::AbstractArray{Float64})
    
    plots_by_jac_noise = []

    for s_J in sigma_jac #for each jacobian noise level

        sigma_results = []
        for s in sigma_noise
            push!(sigma_results,PPR.est_stochastic_bus_voltage_phase!(network,sigma_noise=s,sigma_jac=s_J))
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
            th_rel_err = round(results["th_rel_err"],digits=3)
            sigma = results["sigma_noise"]
            yerr = 2*sqrt.(th_sq_errs)
            plot!(
                θ_hat,
                label=L"$\hat{\theta}_i$ $\sigma =$"*string(sigma*100)*"%, rel_err="*string(th_rel_err)*"%",
                line=:dot,
                marker=:square,
                ribbon=yerr,
                ms=2.5,
                lw=1.5,
                fillalpha=0.15,
                alpha=0.6,
                
            )
        end
        ylabel!(L"Voltage phase angle $\theta_i$ (radians)")
        title!(L"$\hat{\theta}$ by noise level "*string(case_name)*" Jac. Noise Level: "*string(s_J*100))
        push!(plots_by_jac_noise,angle_fig)
    end
    
    p_tot = plot(plots_by_jac_noise,layout=(3,1))
    xlabel!(L"Bus Index $i$")
    
    savefig(p_tot,angle_est_figure_path*"angle_est_"*string(case_name)*".pdf")
    return angle_fig
end

#---- for each jacobian noise level, plot the phase angle noise levels 
#Plot the phase estimates for all sigma noise levels
for (name,path) in zip(network_names,network_paths)
    net = make_basic_network(parse_file(path))
    plot_estimated_angles(net,sigma_noise,sigma_jac)
end