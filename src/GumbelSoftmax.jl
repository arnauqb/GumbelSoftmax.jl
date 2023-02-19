module GumbelSoftmax

using CUDA
using ChainRulesCore
using Flux
using Zygote

include("gumbel_softmax.jl")

end
