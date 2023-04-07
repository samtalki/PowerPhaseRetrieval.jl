using Plots,LaTeXStrings

function plot_avg_vs_inst_tseries(est_averaged_data::Vector,act_average_data::Vector,realtime_data::Vector)
    
    # check periodic properties
    T = length(realtime_data) # total number of data points
    @assert length(est_averaged_data) == length(act_average_data) == length(realtime_data) == T

    # plot the data
    p = plot(
        1:T,
        est_averaged_data,
        legend=true,
        label=L"$\hat{\theta}$ (15m)",
        lw=2,
        la=0.5
    )
    plot!(
        1:T,
        act_average_data,
        label=L"$\bar{\theta}$ (15 min)",
        lw=2,
        la=0.5,
        ls=:dot
    )
    plot!(
        1:T,
        realtime_data,
        label=L"$\theta_t$ (5 min)",
        seriestype=:scatter,
        mcolor=:black,
        ms=2,
        ma=0.5,
    )
    xlabel!(p,L"$t$ (5min)")
    ylabel!(p,L"$\theta$ (rad)")
    return p
end