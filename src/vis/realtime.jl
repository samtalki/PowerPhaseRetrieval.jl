using Plots,LaTeXStrings

function plot_avg_vs_inst_tseries(averaged_data::Vector,realtime_data::Vector)
    
    # check periodic properties
    T = length(realtime_data) # total number of data points
    period = Integer(length(realtime_data)/length(averaged_data)) # number of data points per period
    @assert period*length(averaged_data) == T
    @assert length(averaged_data) == length(realtime_data)/period

    # plot the data
    p = plot(
        1:T,
        averaged_data,
        legend=false,
        label=L"\bar{\theta} (hourly)",
        lw=2 
    )
    plot!(
        1:T,
        realtime_data,
        legend=false,
        label="\theta_t (5 min)",
        seriestype=:scatter,
        ls=:dash,
        ms=1,
        ma=0.5,
    )
    return p
end