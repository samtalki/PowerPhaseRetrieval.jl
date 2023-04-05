include("../src/PowerPhaseRetrieval.jl")
include("../src/core/gershdisc_blocks.jl")
include("../src/core/gershdisc.jl")
import .PowerPhaseRetrieval as PPR
import PowerModels as PM

# 
path = "/home/sam/Research/PowerPhaseRetrieval/data/networks/"
cases = readdir(path)

# Test the Gershgorin disc theorems
case_results = Dict()
phase_recoverable = Dict()
strong_phase_recoverable = Dict()
pct_phase_recoverable = Dict()
pct_strong_phase_recoverable = Dict()
jac_recoverable = Dict()

for case in cases
    # load case and run power flow
    net = PM.make_basic_network(PM.parse_file(path*case))
    if case !== "ACTIVSg2000.m" && case !== "case_ACTIVSg500.m"
        PM.compute_ac_pf!(net)
        println("skip acpf: $case")
    end
    
    # make a results dict for this case
    case_results[case] = Dict()

    # check if the voltage phase angles are observable
    dpξ,dqξ = calc_augmented_angle_jacobians(net)
    case_results[case]["phase_recoverable"] = dpdth_observability(net)
    case_results[case]["dpξ"] = dpξ
    case_results[case]["dqξ"] = dqξ
    case_results[case]["jac_recoverable"] = is_jacobian_block_invertible(net)

    # save the recoverabilities
    phase_recoverable[case] = case_results[case]["phase_recoverable"].observable
    strong_phase_recoverable[case] = case_results[case]["phase_recoverable"].strong_observable
    pct_phase_recoverable[case] = sum(phase_recoverable[case])/length(phase_recoverable[case])
    pct_strong_phase_recoverable[case] = sum(strong_phase_recoverable[case])/length(strong_phase_recoverable[case])
    jac_recoverable[case] = case_results[case]["jac_recoverable"]

end