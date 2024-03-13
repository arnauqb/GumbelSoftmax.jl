using ForwardDiff
using GumbelSoftmax
using Test
using Zygote
using Statistics

@testset "Test conditional" begin
    logits = [[1.0 1.5 0.8]; [0.3 0.2 0.5]]
    D = [[1 0 0]; [0 1 0]]
    k = 2
    n_samples = 10000
    pytorch_truth = permutedims(cat([
            1.8225 -0.0950 0.1325;
            0.0373 1.8112 -0.0229], [
            1.8258 -0.0958 0.1309;
            0.0409 1.8185 -0.0221], dims = 3), [3, 1, 2])
    samples = [GumbelSoftmax.sample_conditional_gumbel(logits, D, k) for i in 1:n_samples]
    mean_sample = mean(samples)
    @test mean_sample ≈ pytorch_truth rtol = 1e-1
end

@testset "Test Rao Gumbel" begin

        
    end
    pytorch_truth = [0.3015, 0.4698, 0.2287]
    pytorch_grad_truth = [-2.2531e-09, -3.7838e-09, -9.6849e-10]
    n_samples = 10000
    logits = [[1.0, 1.5, 0.8]]
    k = 100
    samples_cond =
        samples = mean([sample_rao_gumbel_softmax(logits, k) for i in 1:n_samples], dims = 1)
    @test samples ≈ pytorch_truth atol = 1e-1
    grad_samples_zygote = mean([Zygote.jacobian(logits -> sum(sample_rao_gumbel_softmax(logits, k)), logits)[1] for i in 1:n_samples], dims = 1)
    grad_samples_fd = mean([ForwardDiff.gradient(logits -> sum(sample_rao_gumbel_softmax(logits, k)), logits) for i in 1:n_samples], dims = 1)
    @test grad_samples_zygote ≈ pytorch_grad_truth atol = 1e-1
    @test grad_samples_fd ≈ pytorch_grad_truth atol = 1e-1
end
