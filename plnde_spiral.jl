using DifferentialEquations, DiffEqFlux, DiffEqSensitivity, Flux
using Optim 
using StatsBase
using Distributions
using BSON
using Random

Random.seed!(1)

include("meshgrid.jl") # meshgrid functions
include("example_dynamics.jl") # definitions of differential equations used for synthetic datasets

# ********************************************************* Model ****************************************************************** #

# Hyperparameters
T = 1001        # time stamps rounded to the nearest 1ms
L = 3           # number of latents assumed by the model
L_true = 3      # true number of latents
D = 150         # number of observations
N = 343         # number of trials
dt = 0.001      # 1-ms bin
κ = 1.          # scale parameter for prior initial value distribution covariance
tspan = (0.0f0,40.0f0) # 40 seconds long
trange = range(tspan[1],tspan[2],length=T)
u0 = Float32[1.; 1.; 1.]

f = FastChain(FastDense(L, 17, swish), FastDense(17,23, swish), FastDense(23,17, swish), FastDense(17,L)) # Feedforward Neural Network
_nn_ode = NeuralODE(f, tspan, AutoTsit5(TRBDF2(autodiff=false)), saveat = trange, reltol=1e-6, abstol=1e-6) # Neural ODE
_logλ_true = FastDense(L_true, D) # mapping from latent space (true) to observations
_logλ = FastDense(L, D) # mapping from latent space (model) to observations
p2 = initial_params(_logλ) # initialize parameters for the mapping

nn_ode = (u, p) -> _nn_ode(u, p[1:length(_nn_ode.p)])
logλ = (u, p) -> _logλ(u, p[length(_nn_ode.p)+1:length(_nn_ode.p)+length(p2)])


# ***************************************************** Generate data ************************************************************** #

# Generate loading matrix and bias for this dataset
C = (rand(D, L_true) .+ 2) .* sign.(randn(D, L_true)) # loading matrix -- low mean firing rates
#C = (rand(D, L_true) .+ 8) .* sign.(randn(D, L_true)) # loading matrix -- high mean firing rates
params_λ_true = [collect(Iterators.flatten(C)); zeros(D)]

# Generate initial values for training dataset
x,y,z = ([range(-0.5,0.5,length=7);],[range(-0.5,0.5,length=7);],[range(-0.5,0.5,length=7);])
X,Y,Z = meshgrid(x,y,z)
u0s = [X[:] Y[:] Z[:]]'

# Generate latent trajectories for training dataset
prob_true = ODEProblem(spiral,u0,tspan)
prob_func(prob,i,repeat) = remake(prob, u0=u0s[:,i])
ens_prob = EnsembleProblem(prob_true, prob_func=prob_func)
z_true_mat = Array(solve(ens_prob, Tsit5(), saveat=trange, trajectories=N)) # latent trajectories [latents x timebins x trials]

# Generate training dataset spikes from discretized intensity
spike_times_disc = zeros(D, T, N)
for i = 1:N; spike_times_disc[:,:,i] = (rand.(Poisson.(dt*exp.(_logλ_true(z_true_mat[:,:,i], params_λ_true))))) .> 0; end
spike_times_disc = spike_times_disc .> 0
spikes = permutedims(spike_times_disc, (1, 3, 2)) # spike trains [neurons x trials x timebins]

# Generate initial values for test dataset
u0s_test = rand(L_true, N) .- 0.5

# Generate latent trajectories for test dataset
prob_func_test(prob,i,repeat) = remake(prob, u0=u0s_test[:,i])
ens_prob_test = EnsembleProblem(prob_true, prob_func=prob_func_test)
z_true_test = Array(solve(ens_prob_test, Tsit5(), saveat=trange, trajectories=N)) # latent trajectories [latents x timebins x trials]

# Generate test dataset spikes from discretized intensity
spike_times_test = zeros(D, T, N)
for i = 1:N; spike_times_test[:,:,i] = (rand.(Poisson.(dt*exp.(_logλ_true(z_true_test[:,:,i], params_λ_true))))) .> 0; end
spike_times_test = spike_times_test .> 0
spikes_test = permutedims(spike_times_test, (1, 3, 2)) # spike trains [neurons x trials x timebins]


# ********************************************************** Loss ****************************************************************** #

# Initialize model parameters before training
θ = Float32[Flux.glorot_uniform(length(_nn_ode.p)); randn(D*(L+1)); randn(L*N); -10 .* ones(L*N)]

function loss_nn_ode(p) # negative of the Evidence Lower BOund (ELBO) in Eq. 13
    u0_m = reshape(p[end-L*N-L*N+1:end-L*N], L, :)
    u0_s = clamp.(reshape(p[end-L*N+1:end], L, :), -1e8, 0)
    u0s = u0_m .+ exp.(u0_s) .* randn(size(u0_s))
    kld = 0.5 .* (N .* L .* log(κ) .- sum(2 .* u0_s) - N .* L .+ sum(exp.(2 .* u0_s)) ./ κ .+ sum(abs2, u0_m) ./ κ)
    z_hat = Array(nn_ode(u0s, p))
    λ_hat = exp.(logλ(reshape(z_hat, L, :), p))
    λ_hat = reshape(λ_hat, D, size(z_hat, 2), size(z_hat, 3))
    Nlogλ = spikes[1:size(λ_hat, 1), 1:size(λ_hat, 2), 1:size(λ_hat, 3)] .* log.(dt.*λ_hat .+ sqrt(eps(Float32)))
    loss = ( sum(dt.*λ_hat .- Nlogλ) .+ kld ) ./ N
    return loss
end

cb = function (p, l)
    println(l)
    return false
end

# Burn-in loss
cb(θ,loss_nn_ode(θ))


# ******************************************************** Training **************************************************************** #

"""
Training PLNDE by "iteratively growing fit". Optimizing both the generative and variational parameters.

References:
[1] Rackauckas et al. (2020). Universal differential equations for scientific machine learning. arXiv.
"""

_nn_ode = NeuralODE(f, (0.0, 4.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 4.], reltol=1e-6, abstol=1e-6)
res1 = DiffEqFlux.sciml_train(loss_nn_ode, θ, ADAM(0.01), cb = cb, maxiters = 300)

_nn_ode = NeuralODE(f, (0.0, 8.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 8.], reltol=1e-6, abstol=1e-6)
res2 = DiffEqFlux.sciml_train(loss_nn_ode, res1.minimizer, ADAM(0.005), cb = cb, maxiters = 400)

_nn_ode = NeuralODE(f, (0.0, 12.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 12.], reltol=1e-6, abstol=1e-6)
res3 = DiffEqFlux.sciml_train(loss_nn_ode, res2.minimizer, ADAM(0.005), cb = cb, maxiters = 400)

_nn_ode = NeuralODE(f, (0.0, 16.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 16.], reltol=1e-6, abstol=1e-6)
res4 = DiffEqFlux.sciml_train(loss_nn_ode, res3.minimizer, ADAM(0.005), cb = cb, maxiters = 400)

_nn_ode = NeuralODE(f, (0.0, 20.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 20.], reltol=1e-6, abstol=1e-6)
res5 = DiffEqFlux.sciml_train(loss_nn_ode, res4.minimizer, ADAM(0.001), cb = cb, maxiters = 400)

_nn_ode = NeuralODE(f, (0.0, 24.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 24.], reltol=1e-6, abstol=1e-6)
res6 = DiffEqFlux.sciml_train(loss_nn_ode, res5.minimizer, ADAM(0.001), cb = cb, maxiters = 400)

_nn_ode = NeuralODE(f, (0.0, 28.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 28.], reltol=1e-6, abstol=1e-6)
res7 = DiffEqFlux.sciml_train(loss_nn_ode, res6.minimizer, ADAM(0.001), cb = cb, maxiters = 400)

_nn_ode = NeuralODE(f, (0.0, 32.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 32.], reltol=1e-6, abstol=1e-6)
res8 = DiffEqFlux.sciml_train(loss_nn_ode, res7.minimizer, ADAM(0.001), cb = cb, maxiters = 600)

_nn_ode = NeuralODE(f, (0.0, 36.0), AutoTsit5(TRBDF2(autodiff=false)), saveat = trange[trange .< 36.], reltol=1e-6, abstol=1e-6)
res9 = DiffEqFlux.sciml_train(loss_nn_ode, res8.minimizer, ADAM(0.001), cb = cb, maxiters = 800)

_nn_ode = NeuralODE(f, tspan, AutoTsit5(TRBDF2(autodiff=false)), saveat = trange, reltol=1e-6, abstol=1e-6)
res10 = DiffEqFlux.sciml_train(loss_nn_ode, res9.minimizer, ADAM(0.001), cb = cb, maxiters = 100)

θ_opt = res10.minimizer

BSON.@save "spiral_fit_train_PLNDE.bson" θ_opt spikes z_true_mat params_λ_true


# ******************************************************** Testing ***************************************************************** #

θ_iv = θ_opt[end-L*N-L*N+1:end]

# Redefine the loss such that the generative parameters are frozen and only the variational parameters are optimized
function loss_nn_ode(p)
    u0_m = reshape(p[end-L*N-L*N+1:end-L*N], L, :)
    u0_s = clamp.(reshape(p[end-L*N+1:end], L, :), -1e8, 0)
    u0s1 = u0_m .+ exp.(u0_s) .* randn(size(u0_s))
    kld = 0.5 .* (N .* L .* log(κ) .- sum(2 .* u0_s) - N .* L .+ sum(exp.(2 .* u0_s)) ./ κ .+ sum(abs2, u0_m) ./ κ)
    z_hat = Array(nn_ode(u0s1, θ_opt))
    λ_hat = exp.(logλ(reshape(z_hat, L, :), θ_opt))
    λ_hat = reshape(λ_hat, D, size(z_hat, 2), size(z_hat, 3))
    Nlogλ = spikes_test[1:size(λ_hat, 1), 1:size(λ_hat, 2), 1:size(λ_hat, 3)] .* log.(dt.*λ_hat .+ sqrt(eps(Float32)))
    loss = ( sum(dt.*λ_hat .- Nlogλ) .+ kld ) ./ N
    return loss
end

# Display the ODE with the initial parameter values - burn in loss
cb(θ_iv,loss_nn_ode(θ_iv))

res11 = DiffEqFlux.sciml_train(loss_nn_ode, θ_iv, ADAM(0.01), cb = cb, maxiters = 100)
res12 = DiffEqFlux.sciml_train(loss_nn_ode, res11.minimizer, ADAM(0.001), cb = cb, maxiters = 450)

θ_opt_test = θ_opt
θ_opt_test[end-L*N-L*N+1:end] = res12.minimizer

BSON.@save "spiral_fit_test_PLNDE.bson" θ_opt_test spikes_test z_true_test params_λ_true
