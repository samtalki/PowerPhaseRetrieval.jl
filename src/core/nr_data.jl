# BSD 3-Clause License

# Copyright (c) 2022, Samuel Talkington
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import PowerModels as PM
import LinearAlgebra as LA

"""
Newton Raphson Power Flow data 
Supports a single iteration and all iterations.
"""
struct NRPFData
    n_bus::Int
    iter::Int #iteration number k
    x::Vector{Vector{Complex}} #Rectangular grid state x∈C^n
    f::Vector{Vector{Complex}} #Rectangular bus mismatches f(x) 
    S_inj::Vector{Vector{Complex}} #Computed injections at every iteration
    L_inj::Vector{Vector{Complex}} #Computed current injections at every iteration
    delta_va::Vector{Vector{Real}} #Change in voltage angle
    delta_vm::Vector{Vector{Real}} #Change in voltage mag
    delta_p::Vector{Vector{Real}} #Change in active power
    delta_q::Vector{Vector{Real}} #Change in reactive power
    jacs::Vector{PowerFlowJacobian}
end

"""
Gets iterative Newton-Raphson power flow data
"""
function calc_nr_pf!(data;tol=1e-4,itr_max=20)

    bus_num = length(data["bus"])
    gen_num = length(data["gen"])

    # Count the number of generators per bus
    gen_per_bus = Dict()
    for (i, gen) in data["gen"]
        bus_i = gen["gen_bus"]
        gen_per_bus[bus_i] = get(gen_per_bus, bus_i, 0) + 1
        # Update set point in PV buses
        if data["bus"]["$bus_i"]["bus_type"] == 2
            data["bus"]["$bus_i"]["vm"] = gen["vg"]
        end
    end
    
    rect_x,rect_f = [],[] #rectangular voltages and complex power injections
    S_inj,L_inj = [],[] #Power and current injections
    vm_deltas,va_deltas = [],[]
    p_mismatches,q_mismatches = [],[]
    jacobians = []
    
    Y = calc_basic_admittance_matrix(data)
    itr = 0

    while itr < itr_max
        # STEP 1: Compute mismatch and check convergence
        V = calc_basic_bus_voltage(data)
        S = calc_basic_bus_injection(data)
        Li = conj(Y*V) #Current injection
        Si = V .* Li
        Δp, Δq = real(S - Si), imag(S - Si)
        push!(rect_x,V)
        push!(S_inj,Si)
        push!(L_inj,Li)
        push!(p_mismatches,Δp)
        push!(q_mismatches,Δq)
        push!(rect_f, Δp .+ Δq.*im)

        if LinearAlgebra.normInf([Δp; Δq]) < tol
            break
        end

        # STEP 2 and 3: Compute the jacobian and update step
        J = calc_basic_jacobian_matrix(data)
        push!(jacobians,calc_jacobian_matrix(data))
        x = J \ [Δp; Δq]
        va,vm = x[1:bus_num],x[bus_num+1:end]
        push!(va_deltas,va)
        push!(vm_deltas,vm)
        
        
        # STEP 4
        # update voltage variables
        for i in 1:bus_num
            bus_type = data["bus"]["$(i)"]["bus_type"]
            if bus_type == 1
                data["bus"]["$(i)"]["va"] = data["bus"]["$(i)"]["va"] + x[i]
                data["bus"]["$(i)"]["vm"] = data["bus"]["$(i)"]["vm"] + x[i+bus_num] * data["bus"]["$(i)"]["vm"] 
            end
            if bus_type == 2
                data["bus"]["$(i)"]["va"] = data["bus"]["$(i)"]["va"] + x[i]
            end
        end
        # update power variables
        for i in 1:gen_num
            bus_i = data["gen"]["$i"]["gen_bus"]
            bus_type = data["bus"]["$bus_i"]["bus_type"]
            num_gens = gen_per_bus[bus_i]
            if bus_type == 2
                data["gen"]["$i"]["qg"] = data["gen"]["$i"]["qg"] - Δq[bus_i] / num_gens # TODO it is ok for multiples gens in same bus?
            else bus_type == 3
                data["gen"]["$i"]["qg"] = data["gen"]["$i"]["qg"] - Δq[bus_i] / num_gens
                data["gen"]["$i"]["pg"] = data["gen"]["$i"]["pg"] - Δp[bus_i] / num_gens
            end
        end
        # update iteration counter
        itr += 1
    end
    if itr == itr_max
        Memento.warn(_LOGGER, "Max iteration limit")
        @assert false
    end
    return NRPFData(
        bus_num,
        itr,
        rect_x,
        rect_f,
        S_inj,
        L_inj,
        va_deltas,
        vm_deltas,
        p_mismatches,
        q_mismatches,
        jacobians
    )
end



# """
# The Newton-Raphson-Phase Retrieval Power Flow Algorithm. Compatible with basic networks.
# """
# function calc_basic_ac_pf_data!(data::Dict{String, Any};tol=1e-4,itr_max=50)
#     n_bus = length(data["bus"])
#     Y = PM.calc_basic_admittance_matrix(data)
#     itr = 0

    
#     rect_x,rect_f = [],[]
#     vm_deltas,va_deltas = [],[]
#     mistaches = []
#     p_mismatches,q_mismatches = [],[]

#     while itr < itr_max

#         #--- STEP 0: Compute grid states and save
#         V = PM.calc_basic_bus_voltage(data) #∈Cⁿ
#         S = PM.calc_basic_bus_injection(data) #∈Cⁿ
#         push!(rect_x,V)
#         push!(rect_f,S)

#         #--- STEP 1: Compute mismatch and check convergence
#         Δx = calc_basic_mismatch(V,S,Y)
        
#         #- Save mismatches
#         Δp,Δq = Δx[1:n_bus],Δx[n_bus+1:end]
#         push!(mistaches,Δx)
#         push!(p_mismatches,Δp)
#         push!(q_mismatches,Δq)

#         # - Check convergence
#         if LinearAlgebra.normInf(Δx) < tol
#             break
#         end
        
#         #--- STEP 2 and 3: Compute the jacobian and update step
#         J = PM.calc_basic_jacobian_matrix(data)
#         vph = J \ Δx
#         va,vm = vph[1:n_bus],vph[n_bus+1:end]
        
#         #- Save changes in grid state
#         push!(va_deltas,va)
#         push!(vm_deltas,vm)
        
#         # STEP 4 and 5
#         # update voltage variables
#         data = update_voltage_state!(data,vph)
#         # update power variables
#         data = update_injection_state!(data,Δx)

#         # update iteration counter
#         itr += 1
#     end
#     if itr == itr_max
#         @assert false "Did not converge, max iteration limit"
#     end
    
#     return NRPFData(
#         n_bus,
#         itr,
#         rect_x,
#         rect_f,
#         vm_deltas,
#         va_deltas,
#         p_mismatches,
#         q_mismatches
#     )
# end