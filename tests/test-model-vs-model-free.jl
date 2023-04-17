# --------
# Test the phase retrieval problem with a known and unknown topology
# --------

include("../src/PowerPhaseRetrieval.jl")
include("../src/core/topology_sensitivities.jl")
import .PowerPhaseRetrieval as PPR
using LinearAlgebra,PowerModels
using Statistics,Random,Distributions
using Plots,LaTeXStrings,ColorSchemes

# ---------------- load case and run power flow
net = make_basic_network(parse_file("data/networks/case_RTS_GMLC.m")) #basic network
pq_buses = PPR.calc_bus_idx_of_type(net,[1])
compute_ac_pf!(net)

# ----------------- get the initial values
v0 = calc_basic_bus_voltage(net)[pq_buses]
vm0 = abs.(calc_basic_bus_voltage(net)[pq_buses])
θ0 = angle.(calc_basic_bus_voltage(net)[pq_buses])
p0 = real.(calc_basic_bus_injection(net)[pq_buses])
q0 = imag.(calc_basic_bus_injection(net)[pq_buses])

# ----------------- get the topology data
Y = calc_basic_admittance_matrix(net)[pq_buses,pq_buses]
G,B = real.(Y),imag.(Y)

# ----------------- get the sensitivities (topology known)
dpθ,dqθ,dpvm,dqvm = calc_topology_sensitivities!(net,sel_bus_types=[1])
J_polar_topology = [
    dpθ dpvm;
    dqθ dqvm
]
y_topology = J_polar_topology*[θ0;vm0]

# ----------------- get the sensitivities (polar)
J_polar = PPR.calc_jacobian_matrix(net,[1])
J_polar_mat = Matrix(PPR.calc_jacobian_matrix(net,[1]).matrix)
x_polar_true = [θ0;vm0]
y_polar_true = J_polar_mat*x_polar_true


# ----------------- check that the topology-free and topology-based align
println("Jacobian error: ",norm(J_polar_mat-J_polar_topology))
println("Output error: ",norm(y_polar_true-y_topology))


# ----------------- Function to perform model-based phase retrieval
function est_model_based_voltage_phase(sigma_noise;p_nom=p0,q_nom=q0,θ0=θ0,vm0=vm0)
    sigma_p = mean(abs.(p_nom[abs.(p_nom) .> 0]))*sigma_noise
    sigma_q = mean(abs.(q_nom[abs.(q_nom) .> 0]))*sigma_noise
    d_p,d_q = Normal(0,sigma_p),Normal(0,sigma_q)
    f_x_obs = J_polar_topology*[θ0;vm0] + [rand(d_p,length(p_nom));rand(d_q,length(q_nom))]
    x_hat = inv(Matrix(J_polar_topology))*f_x_obs
    return Dict(
        "th_hat"=>x_hat[1:length(θ0)],
        "th_rel_err"=>norm(x_hat[1:length(θ0)]-θ0)/norm(θ0)*100,
    )    
end



#---------------------- Create and solve the model and plot results
#--- Global plotting parameters

figure_path = "figures/with-without-topology/"
network_folder = "data/networks/"
network_names = ["case24_ieee_rts","case_RTS_GMLC"]
network_paths = [network_folder*net_name*".m" for net_name in network_names]
color_scheme = color_list(:tab10)


#--- Global noise parameters
sigma_jac = [0.05,0.1,0.2]
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
            s_model_based = est_model_based_voltage_phase(s) # ---- only noise in the observations for the model-based case
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

            # ---- plot the model based results
            n = length(mf_θ_hat)
            mb_θ_hat = mb_result["th_hat"]
            th_rel_err = round(mb_result["th_rel_err"],digits=3) # relative error in percent
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
        dpv_err,dqv_err = round(sigma_model_free_results[1]["obs_dpv_rel_err"],digits=3),round(sigma_model_free_results[1]["obs_dqv_rel_err"],digits=3)
        title!(string(case_name)*": "*L"$\hat{\theta}$ with Jac. rel. err.="*string(dpv_err)*"%")
        push!(plots_by_jac_noise,angle_fig)
    end
    p_tot = plot(plots_by_jac_noise...,layout=(length(plots_by_jac_noise),1))
    xlabel!(L"Bus Index $i$")
    
    savefig(p_tot,figure_path*"_mb_v_mf_noisy_"*string(case_name)*".pdf")
end



# --- plot the model based results
