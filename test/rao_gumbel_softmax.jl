using ForwardDiff
using GumbelSoftmax
using Flux
using Test
using Zygote
using Statistics

@testset "Test conditional" begin
    logits = [[1.0 1.5 0.8]; [0.3 0.2 0.5]]
    D = [[1 0 0]; [0 1 0]]
    k = 2
    n_samples = 10000
    pytorch_truth = permutedims(cat(
            [2.8274  1.3816  0.9552;
                0.3226  2.0021  0.4696],
            [2.8097  1.4182  0.9312;
                0.3440  2.0176  0.4887], dims = 3), [3, 1, 2])
    samples = [GumbelSoftmax.sample_conditional_gumbel(logits, D, k) for i in 1:n_samples]
    mean_sample = mean(samples)
    @test mean_sample ≈ pytorch_truth rtol = 1e-1
end

@testset "Test Rao Gumbel" begin
    n_samples = 10000
    logits = [[1.0 1.5 0.8]; [0.3 0.2 0.5]]
    logits = reshape(logits, size(logits)..., 1)
    probs = softmax(logits, dims = 2)
    k = 1000
    samples = mean([sample_rao_gumbel_softmax(logits = logits, k = k) for i in 1:n_samples], dims = 1)[1]
    # we should recover categorical probs
    @test samples ≈ probs rtol = 5e-2

    # gradient
    pytorch_jacobian_truth_1 = [[0.2043 -0.1365 -0.0678];
        [-0.1365 0.2473 -0.1108];
        [-0.0678 -0.1108 0.1786]]
    pytorch_jacobian_truth_2 = [[0.2161 -0.0927 -0.1233];
        [-0.0927 0.2053 -0.1126];
        [-0.1233 -0.1126 0.2359]]
    n_grad_samples = 1000
    logits = [[1.0 1.5 0.8];]
    logits = reshape(logits, size(logits)..., 1)
    grad_samples_zygote = mean([Zygote.jacobian(logits -> sample_rao_gumbel_softmax(logits = logits, k = k, tau = 0.1), logits)[1] for i in 1:n_grad_samples], dims = 1)[1]
    grad_samples_fd = mean([ForwardDiff.jacobian(logits -> sample_rao_gumbel_softmax(logits = logits, k = k, tau = 0.1), logits) for i in 1:n_grad_samples], dims = 1)[1]
    @test grad_samples_zygote ≈ pytorch_jacobian_truth_1 rtol = 1e-1
    @test grad_samples_fd ≈ pytorch_jacobian_truth_1 rtol = 1e-1
    logits = [[0.3 0.2 0.5];]
    logits = reshape(logits, size(logits)..., 1)
    grad_samples_zygote = mean([Zygote.jacobian(logits -> sample_rao_gumbel_softmax(logits = logits, k = k, tau = 0.1), logits)[1] for i in 1:n_grad_samples], dims = 1)[1]
    grad_samples_fd = mean([ForwardDiff.jacobian(logits -> sample_rao_gumbel_softmax(logits = logits, k = k, tau = 0.1), logits) for i in 1:n_grad_samples], dims = 1)[1]
    @test grad_samples_zygote ≈ pytorch_jacobian_truth_2 rtol = 1e-1
    @test grad_samples_fd ≈ pytorch_jacobian_truth_2 rtol = 1e-1
end
