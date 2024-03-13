module GumbelSoftmax

using CUDA
using ChainRulesCore
using Distributions
using Flux
using ForwardDiff
using OneHotArrays
using Random
using Zygote

include("utils.jl")
include("gumbel_softmax.jl")
include("rao_gumbel_softmax.jl")

end
