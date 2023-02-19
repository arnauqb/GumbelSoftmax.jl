using Test
using GumbelSoftmax, Zygote

N = 1000
p = [[0.1, 0.2, 0.3, 0.4] [0.5, 0.6, 0.1, 0.3] [0.4, 0.2, 0.6, 0.3]]


@testset "Test forward behaviour" begin
    function sample_cat(p, N, tau, hard)
        res = zeros(size(p))
        for i = 1:N
            res += sample_gumbel_softmax(p, tau, hard = hard)
        end
        return res / N
    end
    for hard in [true, false]
        for tau in [0.001, 0.01, 0.1]
            res = sample_cat(p, N, tau, hard)
            @test res ≈ p rtol = 0.05
        end
    end
    # for tau >= 1.0 and hard = false, the results are more biased
    res = sample_cat(p, N, 0.1, false)
    @test res ≈ p rtol = 0.1
end


@testset "Test gradient" begin

    function get_total(p, N, tau)
        res = sum([sum(sample_gumbel_softmax(p, tau)[:,2]) for i in 1:N])
        return res
    end
    tau = 0.1
    res = get_total(p, N, tau)
    @test res ≈ 1500 rtol = 0.1

    function get_gradient_estimation(n_gradients, tau)
        gradient_estimation = zeros(size(p))
        for i = 1:n_gradients
            gradient_estimation += gradient(x -> get_total(x, N, tau), p)[1]
        end
        return gradient_estimation / n_gradients
    end

    n_gradients = 100

    tau = 0.1
    result_pytorch = [[-513.5213, -596.2949, -100.7937, -300.1013] [498.3330, 400.7059, 913.0403, 699.4238] [-494.5359, -605.8228, -101.7766, -299.2887]]
    gradient_estimation = get_gradient_estimation(n_gradients, tau)
    @test gradient_estimation ≈ result_pytorch rtol=0.1

    tau = 1.0
    result_pytorch = [[-483.5320, -427.3559, -120.3069, -201.5042] [ 338.8372,  284.9475,  916.4451,  493.3838] [-302.6633, -427.4868,  -92.5874, -224.7113]]
    gradient_estimation = get_gradient_estimation(n_gradients, tau)
    @test gradient_estimation ≈ result_pytorch rtol=0.1

    tau = 0.01
    result_pytorch = [[-500.0630, -568.3240,  -87.3544, -292.3041] [ 505.0536,  401.1935,  859.7476,  702.5405] [-506.3012, -635.2565,  -99.6141, -312.8018]]
    gradient_estimation = get_gradient_estimation(n_gradients, tau)
    @test gradient_estimation ≈ result_pytorch rtol=0.1
end

