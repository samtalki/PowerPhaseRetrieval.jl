using Plots,LaTeXStrings

function plot_avg_vs_inst_tseries(est_averaged_data::Vector,act_average_data::Vector,realtime_data::Vector)
    
    # check periodic properties
    T = length(realtime_data) # total number of data points
    period = Integer(length(realtime_data)/length(est_averaged_data)) # number of data points per period
    @assert period*length(est_averaged_data) == T
    @assert length(est_averaged_data) == length(realtime_data)/period

    # plot the data
    p = plot(
        1:T,
        est_averaged_data,
        legend=true,
        label=L"$\bar{\theta}$ (hourly)",
        lw=2
    )
    plot!(
        1:T,
        act_average_data,
        label=L"$\bar{\theta}$ (5 min)",
        lw=2,
        la=0.5,
        ls=:dot
    )
    plot!(
        1:T,
        realtime_data,
        label=L"$\theta_t$ (5 min)",
        seriestype=:scatter,
        ls=:dash,
        ms=1,
        ma=0.5,
    )
    xlabel!(p,L"$t$ (5min)")
    ylabel!(p,L"$\theta$ (rad)")
    return p
end