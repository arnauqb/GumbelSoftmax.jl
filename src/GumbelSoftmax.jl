module GumbelSoftmax

using CUDA
using ChainRulesCore
using Flux
using ForwardDiff
using Random
using Zygote

include("gumbel_softmax.jl")

end
