module GumbelSoftmax

using CUDA
using ChainRulesCore
using Distributions
using ForwardDiff
using NNlib
using OneHotArrays
using Random
using SliceMap
using Zygote

include("utils.jl")
include("gumbel_softmax.jl")
include("rao_gumbel_softmax.jl")

end
