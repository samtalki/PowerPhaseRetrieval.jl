using Statistics,Distributions

struct NRPRData
    f_x_nom::Vector # f_x_nom = [p_nom;q_nom]
    x_nom::Vector
    J_nom::PowerFlowJacobian #Nominal Jacobian Matrix
end

function NRPFData(network;sel_bus_types=[1])
    #----- Get relevant PQ bus indeces
    compute_ac_pf!(network)
    sel_bus_idx = calc_bus_idx_of_type(network,sel_bus_types)
    n_bus = length(sel_bus_idx) #Get num_bus
    # ---------- Compute nominal ground truth values/params
    #-Compute ground truth complex power injections
    rect_s_nom = calc_basic_bus_injection(network)[sel_bus_idx]
    p_nom,q_nom = real.(rect_s_nom),imag.(rect_s_nom)
    f_x_nom = [p_nom;q_nom]
    #-Compute ground truth voltages
    v_rect_nom = calc_basic_bus_voltage(network)[sel_bus_idx]
    vm_nom,va_nom = abs.(v_rect_nom),angle.(v_rect_nom)
    x_nom = [va_nom;vm_nom]
    #-Compute ground truth Jacobians
    J_nom = calc_jacobian_matrix(network,sel_bus_types) #PQ buses only
    return NRPFData(f_x_nom,x_nom,J_nom_model)
end

"""
Struct for estimated voltage power sensitivities
"""
struct EstimatedVoltagePowerSens
    F_X_obs::AbstractMatrix #Matrix whose columns are random 
    X_obs::Matrix
    dist_loads::Vector{Gamma} #gamma distribution loads
    shape::Union{Vector{Float64},Float64} #Shape params (means of nominal injections)
end

function EstimatedVoltagePowerSens(network::String;sel_bus_types=[1])
    data = NRPFData(network,sel_bus_types)
    shape = [f_x_nom_i for f_x_nom_i in data.f_x_nom]
    dist_load_perturbation = [Gamma(shape_i) for shape_i in shape]
    
    """
    Given a network data dict `data`, and vector of voltage phase angles and voltage magnitudes `vph`, 
    Do: Update the network voltage variables
    """
    function update_voltage_state!(data::Dict{String,<:Any},vph::AbstractVector)
        n_bus = length(data["bus"])
        #@assert length(vph) === 2*n_bus "Voltage phasor solution not equal to number of buses"
        for i in 1:n_bus
            bus_type = data["bus"]["$(i)"]["bus_type"]
            if bus_type == 1
                data["bus"]["$(i)"]["va"] = data["bus"]["$(i)"]["va"] + vph[i]
                data["bus"]["$(i)"]["vm"] = data["bus"]["$(i)"]["vm"] + vph[i+n_bus] * data["bus"]["$(i)"]["vm"] 
            end
            if bus_type == 2
                data["bus"]["$(i)"]["va"] = data["bus"]["$(i)"]["va"] + vph[i]
            end
        end
        return data
    end

end


struct RandomNRPRModel
    n_bus::Int
    sigma_noise::Union{Real,Float64}
    sel_bus_types::Union{AbstractArray{Integer},Integer}
end
