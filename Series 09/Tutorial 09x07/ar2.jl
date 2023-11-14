########################################
# Bayesian Autoregressive AR(2) Model
########################################

#=

References:

# ritvikmath. "Bayesian Time Series : Time Series Talk". 2021. (YouTube)
https://youtu.be/mu-l-K8-8jA

# ritvikmath. Python code for ritvikmath YouTube video.
https://github.com/ritvikmath/YouTubeVideoCode/blob/main/Bayesian%20Time%20Series.ipynb

# "TypeError when sampling AR(2) process using Turing.jl". 2021. Stack Overflow.
https://stackoverflow.com/questions/67730294/typeerror-when-sampling-ar2-process-using-turing-jl

=#

# load packages

using StatsPlots, Turing

using Random;
Random.seed!(2);

# generate data

true_phi_1 = -0.2

true_phi_2 = 0.5

true_sigma = 0.1

time = 50

X = Vector{Float64}(undef, time + 2)

X[1:2] = rand(Normal(0, true_sigma), 2) # Initialize first 2 periods

# is there a way to do this dynamic assignment in a vectorized way, or is this loop the only appraoch?
for t in 3:(time+2)
    noise = rand(Normal(0, true_sigma))
    X[t] = true_phi_1 * X[t-1] +
           true_phi_2 * X[t-2] +
           noise
end

X_data = X[3:end]

# visualize data

p_data = plot(X_data,
    legend=false,
    linewidth=2,
    xlims=(0, 60),
    ylims=(-0.6, 0.6),
    title="Bayesian Autoregressive AR(2) Model - Data",
    xlabel="t",
    ylabel="X_t",
    widen=true
)

########################################
# Model
########################################

# define model

@model function mymodel(time, X)
    # prior
    phi_1 ~ Normal(0, 1)
    phi_2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    # likelihood (must be established for each period's observation in time series)
    # Separate equations are specified in Turing for the likelihood of each observation 
    # (x(t) at each time), but we use the identical equation for each
    X[1] ~ Normal(0, sigma)
    X[2] ~ Normal(0, sigma)
    for t in 3:(time+2)
        mu = phi_1 * X[t-1] + phi_2 * X[t-2]
        X[t] ~ Normal(mu, sigma)
    end
end

# infer posterior probability

model = mymodel(time, X)

sampler = NUTS()

samples = 1_000

chain = sample(model, sampler, samples)

chain
# visualize results

plot(chain)

########################################
# Predictions
########################################

# make predictions
# one separate prediction of the time series evolution over `time_fcst` periods,
#  for each of the chain "samples", each of which is a set of parameter values
#  sampled from posterior distribution estimated by the chain

time_fcst = 10

X_fcst = Matrix{Float64}(
    undef, time_fcst + 2, samples
)

X_fcst[1, :] .= X_data[time-1]

X_fcst[2, :] .= X_data[time]

for col in 1:samples # call the posterior sample parameter values the forecast "fcst" values
    phi_1_fcst = rand(chain[:, 1, 1]) # chain is indexed by (sample, parameter, chainnumber)
    phi_2_fcst = rand(chain[:, 2, 1]) # this seems odd: randomly selecting from different chain samples
    error_fcst = rand(chain[:, 3, 1])
    noise_fcst = rand(Normal(0, error_fcst))
    for row in 3:(time_fcst+2)
        X_fcst[row, col] =
            phi_1_fcst * X_fcst[row-1, col] +
            phi_2_fcst * X_fcst[row-2, col] +
            noise_fcst
    end
end

# visualize predictions

ts_fcst = time:(time+time_fcst) # times assoc with the forecast values

for i in 1:samples
    plot!(p_data, ts_fcst, X_fcst[2:end, i],
        legend=false,
        # predictions
        linewidth=1, color=:green, alpha=0.1
    )
end

p_data

# visualize mean values for predictions

X_fcst_mean = [
    mean(X_fcst[i, 1:samples]) for i in 2:(time_fcst+2)
]

plot!(p_data, ts_fcst, X_fcst_mean,
    legend=false,
    linewidth=2,
    color=:red,
    linestyle=:dot
)

# visualize std dev values for predictions

X_fcst_std = [
    std(X_fcst[i, 1:samples]) for i in 2:(time_fcst+2)
]

plot!(p_data, ts_fcst, [X_fcst_mean .+ 2 * X_fcst_std, X_fcst_mean .- 2 * X_fcst_std],
    legend=false,
    linewidth=2,
    color=:green,
    linestyle=:dot
)

# ====================

# Repeat process, but with a different way to sample from the chain for forecasting
p_data2 = plot(X_data,
    legend=false,
    linewidth=2,
    xlims=(0, 60),
    ylims=(-0.6, 0.6),
    title="Bayesian Autoregressive AR(2) Model - Data",
    xlabel="t",
    ylabel="X_t",
    widen=true
)

# different way to forecast, using sets of parameter values each from same iteration of chain (consistently estimated)
for col in 1:samples # call the posterior sample parameter values the forecast "fcst" values
    phi_1_fcst = chain[col, 1, 1] # chain is indexed by (sample, parameter, chainnumber)
    phi_2_fcst = chain[col, 2, 1] # this approach is much less noisy forecast than randomly selecting from different chain samples
    error_fcst = chain[col, 3, 1]
    noise_fcst = rand(Normal(0, error_fcst))
    for row in 3:(time_fcst+2)
        X_fcst[row, col] =
            phi_1_fcst * X_fcst[row-1, col] +
            phi_2_fcst * X_fcst[row-2, col] +
            noise_fcst
    end
end

# visualize predictions

ts_fcst = time:(time+time_fcst) # times assoc with the forecast values

for i in 1:samples
    plot!(p_data2, ts_fcst, X_fcst[2:end, i],
        legend=false,
        # predictions
        linewidth=1, color=:green, alpha=0.1
    )
end

p_data2

# visualize mean values for predictions

X_fcst_mean = [
    mean(X_fcst[i, 1:samples]) for i in 2:(time_fcst+2)
]

plot!(p_data2, ts_fcst, X_fcst_mean,
    legend=false,
    linewidth=2,
    color=:red,
    linestyle=:dot
)

# visualize std dev values for predictions

X_fcst_std = [
    std(X_fcst[i, 1:samples]) for i in 2:(time_fcst+2)
]

plot!(p_data2, ts_fcst, [X_fcst_mean .+ 2 * X_fcst_std, X_fcst_mean .- 2 * X_fcst_std],
    legend=false,
    linewidth=2,
    color=:green,
    linestyle=:dot
)