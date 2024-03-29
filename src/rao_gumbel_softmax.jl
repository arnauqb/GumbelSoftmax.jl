# This is a direct port from this code:
# https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py

export sample_rao_gumbel_softmax

function sample_conditional_gumbel(logits, D, k = 1)
	# E should be size (k, n_samples, n_classes)
	E = rand(Exponential(1), k, size(logits)...)
	# Ei should be size (k, n_samples, 1)
	D = reshape(D, 1, size(D)...)
	Ei = sum(D .* E, dims = 3)
	# Z should be size (n_samples, 1)
	Z = sum(exp.(logits), dims = 2)
	logits = reshape(logits, 1, size(logits)...)
	Z = reshape(Z, 1, size(Z)...)
	adjusted = D .* (-log.(Ei) .+ log.(Z)) .+ (1 .- D) .* (-log.(E ./ exp.(logits) .+ Ei ./ Z))
	return (logits .- stop_gradient(logits)) .+ stop_gradient(adjusted)
end

function sample_rao_gumbel_softmax_no_batch(; probs = nothing, logits = nothing, k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
	if logits === nothing
		logits = log.(probs .+ epsilon)
	end
	if probs === nothing
		probs = softmax(logits, dims = 2)
	end
	num_classes = size(logits, 2)
	if I === nothing
		# note: calling value here beacuse it can interfere with StochasticAD.jl derivative of the Categorical.
		I = [rand(Categorical(ForwardDiff.value.(probs[i, :]))) for i in axes(probs, 1)]
	end
	D = hcat(onehot.(I, Ref(1:num_classes))...)'
	adjusted = sample_conditional_gumbel(logits, D, k)
	surrogate = mean(softmax(adjusted ./ tau, dims = 3), dims = 1)
	return ((surrogate-stop_gradient(surrogate))+stop_gradient(reshape(D, 1, size(D)...)))[1, :, :]
end

"""
	sample_rao_gumbel_softmax(; probs=nothing, logits=nothing, k=1, tau=1.0, I=nothing, epsilon=1e-10)

Sample from the Rao-Blackwellized Gumbel-Softmax distribution. See https://arxiv.org/abs/2010.04838 for more details.
The expected inputs are either `probs` or `logits`. If `logits` is not provided, it is computed as `log(probs + epsilon)` where `epsilon`
is a small value to avoid numerical instability. The expected shape of `logits` is `(latent_dimension, categorial_dimension, batch_dimension)` to
keep it consistent with Flux. `tau` is a temperature parameter that controls the smoothness of the distribution. `k` is the number of Monte Carlo samples.
"""
function sample_rao_gumbel_softmax(; probs = nothing, logits = nothing, k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
	return sample_rao_gumbel_softmax_no_batch.(probs = probs, logits = logits, k = k, tau = tau, I = I, epsilon = epsilon)
end
