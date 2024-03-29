using GumbelSoftmax
using Test

@testset "GumbelSoftmax.jl" begin
    include("gumbel_softmax.jl")
    include("rao_gumbel_softmax.jl")
end
