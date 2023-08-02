using CUDA, GumbelSoftmax
using Test


@testset "test in gpu" begin
    probs = cu([[0.1, 0.2, 0.3, 0.4] [0.5, 0.6, 0.1, 0.3] [0.4, 0.2, 0.6, 0.3]])
    N = 10000
    function sample_cat(p, N, tau, hard)
        res = sample_gumbel_softmax(p, tau, hard = hard)
        for i = 1:N-1
            res += sample_gumbel_softmax(p, tau, hard = hard)
        end
        return res / N
    end
    for hard in [true, false]
        for tau in [0.001, 0.01, 0.1]
            res = sample_cat(probs, N, tau, hard)
            @test res ≈ probs rtol = 0.05
            @test typeof(res) <: CuArray
        end
    end
    # for tau >= 1.0 and hard = false, the results are more biased
    res = sample_cat(p, N, 0.1, false)
end